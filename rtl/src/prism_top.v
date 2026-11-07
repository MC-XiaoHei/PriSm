// One forecast, end to end: normalise, fold onto the lattice, project in, run
// the mixing block twice, project out, add the linear shortcut.
//
//   y = out_proj(block(block(in_proj(fold(x))))) + skip(fold(x))
//
// The lattice never lands in a wide scratch array: the input projection hands
// one value per cycle straight to the block's load port, and the read-back port
// feeds the output projection the same way. Storing it in registers instead
// cost more logic than the block itself.
`timescale 1ns / 1ps

module prism_top #(
    parameter integer L   = 96,
    parameter integer P   = 24,
    parameter integer HP  = 4,
    parameter integer D   = 16,
    parameter integer W   = 8,
    parameter integer KL  = 5,
    parameter integer R   = 1,
    parameter integer Q   = 12,
    parameter integer WQ  = 14,
    parameter integer NPASS = 2
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        ld_en,
    input  wire [7:0]  ld_addr,
    input  wire [15:0] ld_data,
    output reg         done,
    output reg  signed [15:0] out_data,
    output reg  [7:0]  out_addr,
    output reg         out_we
);
    localparam integer LP   = L / P;         // values per phase row
    localparam integer NC   = P * D;         // cells in the lattice
    localparam integer NW   = NC / W;        // words in the lattice
    localparam integer NP   = HP * P;        // outputs
    localparam integer FLR  = (5 << Q) / 100;

    localparam [3:0] S_IDLE = 0, S_LD = 1, S_SUM = 2, S_D1 = 3, S_D2 = 4,
                     S_D3 = 5, S_D4 = 6, S_Q1 = 7, S_Q2 = 8, S_INV = 9,
                     S_FOLD = 10, S_IN = 11, S_RUN = 12, S_OUT = 13,
                     S_SKIP = 14, S_EMIT = 15, S_DONE = 16;

    // ---- window and folded lattice --------------------------------------
    reg signed [15:0] xm [0:L-1];
    // Every address handed to this memory is (t/D)*LP + i with i < LP, so the
    // lattice is addressed through P*LP words rather than all NC cells.
    reg signed [15:0] zk [0:P*LP-1];
    reg signed [15:0] ym [0:NP-1];

    reg [7:0]  st;
    reg [15:0] t;          // generic counter
    reg [15:0] i;          // inner counter
    reg [15:0] ldw;        // word index while streaming into the block

    // ---- restoring divider ----------------------------------------------
    reg  [31:0] dv_arg, dv_div, dv_n, dv_d, dv_q, dv_r;
    reg  [5:0]  dv_bit;
    reg         dv_go, dv_done, dv_neg, dv_run;

    always @(posedge clk) begin
        if (!rst_n) begin
            dv_q <= 0; dv_r <= 0; dv_done <= 0; dv_run <= 0; dv_bit <= 6'd32;
        end else begin
            dv_done <= 1'b0;
            if (dv_go) begin
                dv_n <= dv_arg; dv_d <= dv_div;
                dv_q <= 0; dv_r <= 0; dv_bit <= 0; dv_run <= 1'b1;
            end else if (dv_bit != 6'd32) begin
                dv_bit <= dv_bit + 1;
                if ({dv_r[30:0], dv_n[31]} >= dv_d) begin
                    dv_r <= {dv_r[30:0], dv_n[31]} - dv_d;
                    dv_n <= {dv_n[30:0], 1'b0};
                    dv_q <= {dv_q[30:0], 1'b1};
                end else begin
                    dv_r <= {dv_r[30:0], dv_n[31]};
                    dv_n <= {dv_n[30:0], 1'b0};
                    dv_q <= {dv_q[30:0], 1'b0};
                end
            end else if (dv_run) begin
                dv_done <= 1'b1; dv_run <= 1'b0;
            end
        end
    end

    // ---- restoring square root ------------------------------------------
    reg  [31:0] sq_arg, sq_x, sq_res, sq_bit;
    reg         sq_go, sq_done;

    always @(posedge clk) begin
        if (!rst_n) begin
            sq_done <= 0; sq_bit <= 0; sq_res <= 0; sq_x <= 0;
        end else begin
            sq_done <= 1'b0;
            if (sq_go) begin
                sq_x <= sq_arg; sq_res <= 0; sq_bit <= 32'h4000_0000;
            end else if (sq_bit != 0) begin
                if (sq_x >= sq_res + sq_bit) begin
                    sq_x   <= sq_x - (sq_res + sq_bit);
                    sq_res <= (sq_res >> 1) + sq_bit;
                end else sq_res <= (sq_res >> 1);
                sq_bit <= sq_bit >> 2;
                if (sq_bit == 32'd1) sq_done <= 1'b1;
            end
        end
    end

    // The product needs its own 32-bit home: written inline, the assignment
    // context squeezes the multiply down to 16 bits before the shift.
    wire signed [31:0] fold_prod =
        ($signed(xm[i]) - $signed(meanw_q[15:0])) * $signed(inv_q[15:0]);

    reg signed [39:0] acc1, acc2, acc1w, acc2w;
    reg signed [31:0] mean_q, meanw_q, ex2, ex2w, std_loc, std_all, std_q, inv_q;

    // Round and clip in one place: an inline shift inherits the 16-bit
    // destination width and squeezes the accumulator first.
    function signed [15:0] sat16;
        input signed [43:0] v;
        reg signed [43:0] u;
        begin
            u = (v + (44'sd1 <<< (WQ-1))) >>> WQ;
            if (u > 44'sd32767) sat16 = 16'sd32767;
            else if (u < -44'sd32768) sat16 = -16'sd32768;
            else sat16 = u[15:0];
        end
    endfunction

    // ---- weights for the projections -------------------------------------
    reg signed [15:0] w_in  [0:LP*D-1];
    reg signed [15:0] w_out [0:D*HP-1];
    reg signed [15:0] w_skip[0:LP*HP-1];
    // Projection biases: one constant per output feature (in) or per horizon (out).
    reg signed [15:0] b_in  [0:D-1];
    reg signed [15:0] b_out [0:HP-1];

    initial begin
        $readmemh("w_in.hex",   w_in);
        $readmemh("w_out.hex",  w_out);
        $readmemh("w_skip.hex", w_skip);
        $readmemh("b_in.hex",   b_in);
        $readmemh("b_out.hex",  b_out);
    end

    // ---- block ------------------------------------------------------------
    reg         bstart, ld_blk;
    reg  [9:0]  ldA, rdA;
    wire        bdone;
    wire        bdbg;
    wire [16*W-1:0] rdW;

    lattice_core_w #(.D(D), .W(W), .LP(P), .KL(KL), .R(R),
                     .WQ(WQ), .ACT_RELU(1), .LD_SERIAL(1), .NPASS(NPASS))
    u_blk (.clk(clk), .rst_n(rst_n), .start(bstart),
           .ld_en(ld_blk), .ld_addr(ldA), .ld_data(ld_val),
           .done(bdone), .dbg(bdbg),
           .rd_addr_e(rdA), .rd_word(rdW));

    // ---- input projection, one value per cycle ---------------------------
    reg signed [15:0] ma, mb;
    wire signed [31:0] prod = ma * mb;
    reg  signed [43:0] macc;
    reg  signed [43:0] dot;
    reg signed [43:0] dotq;

    always @(*) begin
        ma = 0; mb = 0;
        case (st)
        S_IN:   begin ma = zk[(t / D) * LP + i]; mb = w_in[i * D + (t % D)]; end
        default: ;
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) macc <= 0;
        else if (st == S_IN) macc <= (i == 0) ? prod : macc + prod;
        else macc <= 0;
    end

    // the input projection is affine, so its bias joins the dot product before
    // the single rounding that takes the accumulator back to Q12
    wire signed [43:0] bin_scaled = $signed({{28{b_in[t % D][15]}}, b_in[t % D]}) <<< 12;

    always @(*) begin
        dot  = macc + prod + bin_scaled;
        dotq = (dot + (44'sd1 <<< (WQ-1))) >>> WQ;
    end

    reg  signed [15:0] ld_val;   // held for the cycle the block samples it

    // ---- output projection, four horizons per lattice cell ---------------
    reg signed [43:0] oacc [0:HP-1];
    reg signed [43:0] sacc [0:HP-1];
    reg [15:0] t_d, t_d2;      // stream counter delayed to the read port
    integer o;

    // The read port answers two cycles after the address is issued, so the
    // multiply runs on the counter as it was two cycles ago.
    wire [15:0] lane_idx = t_d2 % W;
    wire signed [15:0] zval = rdW[lane_idx*16 +: 16];

    always @(posedge clk) begin
        if (!rst_n) begin
            for (o = 0; o < HP; o = o + 1) begin oacc[o] <= 0; sacc[o] <= 0; end
        end else if (st == S_OUT) begin
            for (o = 0; o < HP; o = o + 1)
                if (t_d2 % D == 0) oacc[o] <= zval * w_out[(t_d2 % D) * HP + o];
                else oacc[o] <= oacc[o] + zval * w_out[(t_d2 % D) * HP + o];
        end
    end

    // ---- sequencer --------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 0;
            t <= 0; i <= 0; ldw <= 0;
            ld_blk <= 0; bstart <= 0; ldA <= 0; rdA <= 0;
            out_we <= 0; out_addr <= 0; out_data <= 0;
            sq_go <= 0; dv_go <= 0; ld_val <= 0;
            acc1 <= 0; acc2 <= 0; acc1w <= 0; acc2w <= 0;
            t_d <= 0; t_d2 <= 0;
        end else begin
            ld_blk <= 1'b0; bstart <= 1'b0; out_we <= 1'b0;
            sq_go <= 1'b0; dv_go <= 1'b0;
            if (ld_en) xm[ld_addr] <= ld_data;

            case (st)
            S_IDLE: if (start) begin
                st <= S_LD; i <= 0; t <= 0; done <= 1'b0;
            end

            S_LD: begin
                st <= S_SUM; i <= 0;
                acc1 <= 0; acc2 <= 0; acc1w <= 0; acc2w <= 0;
            end

            S_SUM: begin
                acc1 <= acc1 + xm[i];
                // signed product: a concatenation would drop the sign and
                // square the two's-complement pattern instead
                acc2 <= acc2 + $signed(xm[i]) * $signed(xm[i]);
                if (i >= L-32) begin
                    acc1w <= acc1w + xm[i];
                    acc2w <= acc2w + $signed(xm[i]) * $signed(xm[i]);
                end
                if (i == L-1) begin
                    st <= S_D1;
                    dv_neg <= (acc1 + xm[i]) < 0;
                    dv_arg <= (acc1 + xm[i]) < 0 ? $unsigned(-(acc1 + xm[i]))
                                                 : $unsigned(acc1 + xm[i]);
                    dv_div <= 32'd96; dv_go <= 1'b1;
                end else i <= i + 1;
            end

            S_D1: if (dv_done) begin
                mean_q <= dv_neg ? -$signed(dv_q) : $signed(dv_q); st <= S_D2;
                dv_neg <= 1'b0;
                dv_arg <= $unsigned(acc2); dv_div <= 32'd96; dv_go <= 1'b1;
            end

            S_D2: if (dv_done) begin
                ex2 <= $signed(dv_q); st <= S_D3;
                dv_neg <= acc1w < 0;
                dv_arg <= acc1w < 0 ? $unsigned(-acc1w) : $unsigned(acc1w);
                dv_div <= 32'd32; dv_go <= 1'b1;
            end

            S_D3: if (dv_done) begin
                meanw_q <= dv_neg ? -$signed(dv_q) : $signed(dv_q); st <= S_D4;
                dv_neg <= 1'b0;
                dv_arg <= $unsigned(acc2w); dv_div <= 32'd32; dv_go <= 1'b1;
            end

            S_D4: if (dv_done) begin
                ex2w <= $signed(dv_q); st <= S_Q1;
                sq_arg <= ($signed(dv_q) > meanw_q * meanw_q)
                          ? ($signed(dv_q) - meanw_q * meanw_q) : 32'd0;
                sq_go <= 1'b1;
            end

            S_Q1: if (sq_done) begin
                std_loc <= sq_res; st <= S_Q2;
                sq_arg <= (ex2 > mean_q * mean_q) ? (ex2 - mean_q * mean_q) : 32'd0;
                sq_go <= 1'b1;
            end

            S_Q2: if (sq_done) begin
                std_all <= sq_res; st <= S_INV;
                // local scale floored against a tenth of the window scale and
                // against 0.05, so a near-flat channel cannot blow up
                std_q <= (sq_res / 10 > std_loc) ? ((sq_res / 10 > FLR) ? sq_res / 10 : FLR)
                       : ((std_loc > FLR) ? std_loc : FLR);
                dv_arg <= 32'h0100_0000;
                dv_div <= (sq_res / 10 > std_loc) ? ((sq_res / 10 > FLR) ? sq_res / 10 : FLR)
                       : ((std_loc > FLR) ? std_loc : FLR);
                dv_go <= 1'b1;
            end

            S_INV: if (dv_done) begin
                inv_q <= dv_q; st <= S_FOLD; i <= 0;
            end

            S_FOLD: begin
                zk[(i % P) * LP + (i / P)] <= fold_prod >>> 12;
                if (i == L-1) begin st <= S_IN; i <= 0; t <= 0; ldw <= 0; end
                else i <= i + 1;
            end

            // one projected value per cycle, straight into the block
            // one projected value per cycle, straight into the block: the load
            // strobe fires only when the running dot product is complete
            S_IN: begin
                if (i == LP-1) begin
                    ld_blk <= 1'b1;
                    ld_val <= dotq[15:0];
                    ldA <= t[9:0] / W;
                    i <= 0;
                    t <= t + 1;
                    if (t == NC-1) begin st <= S_RUN; bstart <= 1'b1; t <= 0; end
                end else i <= i + 1;
            end

            S_RUN: if (bdone) begin st <= S_OUT; t <= 0; t_d <= 0; t_d2 <= 0; end

            // 384 cells, one per cycle, four horizons accumulated in parallel
            S_OUT: begin
                rdA <= t[9:0] / W;
                t_d <= t; t_d2 <= t_d;
                if (t_d2 % D == D-1)
                    for (o = 0; o < HP; o = o + 1)
                        ym[(t_d2 / D) * HP + o] <=
                            sat16(oacc[o] + zval * w_out[(t_d2 % D) * HP + o]
                                  + ($signed({{28{b_out[o][15]}}, b_out[o]}) <<< 12));
                // the exit must not be followed by an increment of the counter
                // the next state starts from
                if (t_d2 == NC-1) begin st <= S_SKIP; t <= 0; i <= 0; end
                else t <= t + 1;
            end

            // linear shortcut: a second, cheaper path from the folded window
            S_SKIP: begin
                for (o = 0; o < HP; o = o + 1)
                    sacc[o] <= (i == 0) ? ($signed(zk[t*LP + i]) * $signed(w_skip[i*HP + o]))
                                        : (sacc[o] + $signed(zk[t*LP + i]) * $signed(w_skip[i*HP + o]));
                if (i == LP-1) begin
                    for (o = 0; o < HP; o = o + 1)
                        ym[t*HP + o] <= ym[t*HP + o]
                            + sat16(sacc[o] + $signed(zk[t*LP + i]) * $signed(w_skip[i*HP + o]));
                    i <= 0; t <= t + 1;
                    if (t == P-1) begin st <= S_EMIT; i <= 0; end
                end else i <= i + 1;
            end

            S_EMIT: begin
                // ym is [phase][horizon]; the reference hands back [horizon][phase]
                out_addr <= (i % HP) * P + (i / HP);
                out_data <= ym[i];
                out_we   <= 1'b1;
                if (i == NP-1) begin st <= S_DONE; done <= 1'b1; end
                else i <= i + 1;
            end

            S_DONE: ;
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
