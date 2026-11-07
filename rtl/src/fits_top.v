// FITS in fixed point: instance-normalise the window, keep the lowest bins,
// transform back, rescale.
//
// The forward and inverse transforms are literal sums over the bins that
// survive the low-pass. 96 has no radix-2 factorisation, so a butterfly network
// would need a mixed-radix (2 and 3) tree and would still have to agree with
// numpy's ordering; the sum is unambiguous and the hardware is one multiply and
// one accumulator.
`timescale 1ns / 1ps

module fits_top #(
    parameter integer L    = 96,
    parameter integer H    = 96,
    parameter integer NBIN = 15,
    parameter integer Q    = 12,
    parameter integer WQ   = 14,
    parameter integer FLR  = (5 << Q) / 100
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
    localparam [3:0] S_IDLE = 0, S_SUM = 1, S_D1 = 2, S_D2 = 3, S_Q1 = 4,
                     S_INV = 5, S_SEQ = 6, S_FRE = 7, S_FIM = 8, S_REC = 9,
                     S_DV = 10, S_EMIT = 11, S_DONE = 12;

    reg  [3:0]  st;
    reg  [15:0] i, k, n;

    reg signed [15:0] xm  [0:L-1];
    reg signed [15:0] seq [0:L-1];
    reg signed [31:0] xre [0:NBIN-1];
    reg signed [31:0] xim [0:NBIN-1];
    reg signed [15:0] yq  [0:L-1];

    reg signed [15:0] cs [0:NBIN*L-1];
    reg signed [15:0] sn [0:NBIN*L-1];

    initial begin
        $readmemh("fits_cos.hex", cs);
        $readmemh("fits_sin.hex", sn);
    end

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

    function signed [15:0] clamp16;
        input signed [63:0] v;
        begin
            if (v > 64'sd32767) clamp16 = 16'sd32767;
            else if (v < -64'sd32768) clamp16 = -16'sd32768;
            else clamp16 = v[15:0];
        end
    endfunction

    // ---- statistics -------------------------------------------------------
    reg signed [63:0] acc1, acc2;
    reg signed [31:0] mean_q, std_q, ex2, inv_q;

    wire signed [31:0] fold_prod = ($signed(xm[i]) - $signed(mean_q[15:0]))
                                 * $signed(inv_q[15:0]);

    // ---- transform --------------------------------------------------------
    reg signed [63:0] acc, tot;
    reg signed [15:0] ma, mb;
    wire signed [31:0] prod = ma * mb;
    wire signed [63:0] acc_now = (n == 0) ? prod : (acc + prod);

    always @(*) begin
        ma = 0; mb = 0;
        case (st)
        S_FRE: begin ma = seq[n]; mb = cs[k*L + n];  end
        S_FIM: begin ma = seq[n]; mb = -sn[k*L + n]; end
        default: ;
        endcase
    end

    // inverse accumulates one bin at a time for the current output sample
    wire signed [63:0] rec_term = 2 * ($signed(xre[k]) * $signed(cs[k*L + n])
                                     - $signed(xim[k]) * $signed(sn[k*L + n]));
    wire signed [63:0] tot_now  = (k == 0) ? ($signed(xre[0]) <<< WQ) : (tot + rec_term);

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 0;
            i <= 0; k <= 0; n <= 0;
            acc1 <= 0; acc2 <= 0; acc <= 0; tot <= 0;
            dv_go <= 0; sq_go <= 0; out_we <= 0; out_addr <= 0; out_data <= 0;
        end else begin
            dv_go <= 1'b0; sq_go <= 1'b0; out_we <= 1'b0;
            if (ld_en) xm[ld_addr] <= ld_data;

            case (st)
            S_IDLE: if (start) begin
                st <= S_SUM; i <= 0; done <= 1'b0;
                acc1 <= 0; acc2 <= 0;
            end

            S_SUM: begin
                acc1 <= acc1 + xm[i];
                acc2 <= acc2 + $signed(xm[i]) * $signed(xm[i]);
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
                dv_arg <= acc2[31:0]; dv_div <= 32'd96; dv_go <= 1'b1;
            end

            S_D2: if (dv_done) begin
                ex2 <= $signed(dv_q); st <= S_Q1;
                sq_arg <= ($signed(dv_q) > mean_q * mean_q)
                          ? ($signed(dv_q) - mean_q * mean_q) : 32'd0;
                sq_go <= 1'b1;
            end

            S_Q1: if (sq_done) begin
                std_q <= (sq_res > FLR) ? sq_res : FLR; st <= S_INV;
                dv_arg <= 32'h0100_0000;
                dv_div <= (sq_res > FLR) ? sq_res : FLR;
                dv_go <= 1'b1;
            end

            S_INV: if (dv_done) begin
                inv_q <= dv_q; st <= S_SEQ; i <= 0;
            end

            S_SEQ: begin
                seq[i] <= fold_prod >>> 12;
                if (i == L-1) begin st <= S_FRE; i <= 0; k <= 0; n <= 0; acc <= 0; end
                else i <= i + 1;
            end

            // forward: one bin per pass, two passes (real then imaginary)
            S_FRE, S_FIM: begin
                acc <= acc_now;
                if (n == L-1) begin
                    if (st == S_FRE) xre[k] <= ($signed(acc_now) + (64'sd1 <<< 11)) >>> 12;
                    else             xim[k] <= ($signed(acc_now) + (64'sd1 <<< 11)) >>> 12;
                    n <= 0;
                    if (k == NBIN-1) begin
                        st <= (st == S_FRE) ? S_FIM : S_REC;
                        k <= 0;
                    end else k <= k + 1;
                end else n <= n + 1;
            end

            // inverse: one output sample at a time, over the surviving bins
            S_REC: begin
                tot <= tot_now;
                if (k == NBIN-1) begin
                    st <= S_DV;
                    // the divider only sees magnitudes; the sign is restored
                    // when the quotient is captured
                    dv_neg <= tot_now < 0;
                    dv_arg <= (tot_now < 0)
                              ? (-(($signed(tot_now) + (64'sd1 <<< 15)) >>> 16))
                              : (($signed(tot_now) + (64'sd1 <<< 15)) >>> 16);
                    dv_div <= 32'd96; dv_go <= 1'b1;
                end else k <= k + 1;
            end

            S_DV: if (dv_done) begin
                yq[n] <= dv_neg ? -$signed(dv_q[15:0]) : $signed(dv_q[15:0]);
                tot <= 0; k <= 0;
                if (n == L-1) begin st <= S_EMIT; i <= 0; end
                else begin st <= S_REC; n <= n + 1; end
            end

            S_EMIT: begin
                out_addr <= i[7:0];
                out_data <= clamp16((($signed(yq[i]) * $signed(std_q)) >>> 12)
                                    + $signed(mean_q));
                out_we   <= 1'b1;
                if (i == H-1) begin st <= S_DONE; done <= 1'b1; end
                else i <= i + 1;
            end

            S_DONE: ;
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
