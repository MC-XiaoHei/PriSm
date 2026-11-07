// Lattice mixing block, parameterised over the feature-axis lane width W.
//
//   h[a][f] = sum_t w_dw[f][t] * z[wrap(a + t - KL/2)][f]      circular, phase axis
//   g[r][f] = sum_a w_v[r][a] * h[a][f]                        rank-R, phase axis
//   h[a][f] = act(h[a][f] + sum_r w_u[a][r] * g[r][f])
//   z[a][fo] += sum_fi w_p[fo][fi] * h[a][fi]
//
// Layout. The lattice is LP rows of D features. One memory word holds W
// consecutive features, so a row occupies BLK = D/W words and the word holding
// (row a, block c) sits at address a*BLK + c. Lane j owns feature c*W + j.
//
// Fixed point: data Q12, weights Q14, one 44-bit accumulator per lane. Every
// product is Q12*Q14 = Q26; the accumulator stays at Q26 and the saturator
// rounds once on the way back to Q12.
//
// Timeline. Stage 0 issues addresses and selects weights, stage 1 multiplies and
// accumulates, stage 2 saturates and writes back. One MAC per lane per cycle.
`timescale 1ns / 1ps

module lattice_core_w #(
    parameter integer D  = 16,
    parameter integer W  = 1,
    parameter integer LP = 24,
    parameter integer KL = 5,
    parameter integer R  = 1,
    parameter integer WQ = 14,
    parameter integer ACT_RELU = 1,
    parameter integer LD_SERIAL = 1,
    parameter integer NPASS = 1
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire         ld_en,
    input  wire [9:0]   ld_addr,
    input  wire [LD_SERIAL ? 15 : 16*W-1:0] ld_data,
    output reg          done,
    output wire         dbg,
    // Read-back port: while the core is idle its z memory can be addressed from
    // outside, which is how a chain of cores passes the lattice along.
    input  wire [9:0]   rd_addr_e,
    output wire [16*W-1:0] rd_word
);
    localparam integer BLK = D / W;
    localparam integer DEP = LP * BLK;
    localparam integer AW  = (DEP <= 2) ? 1 : $clog2(DEP);
    localparam integer WW  = 16 * W;
    // power-of-two strides keep the weight index free of multipliers
    localparam integer PAD_KL = 1 << $clog2(KL);
    localparam integer PAD_LP = 1 << $clog2(LP);

    localparam [3:0] S_IDLE = 0, S_DW = 1, S_DOWN = 2, S_UP = 3, S_POINT = 4,
                     S_DONE = 5;

    // ---- weights ---------------------------------------------------------
    // The two passes carry distinct weights, so every table holds two banks and
    // `pass` selects one. Bank 0 lives in the original file, bank 1 in the _1
    // file loaded into the upper half of the same array.
    localparam integer B_DW = D*PAD_KL;
    localparam integer B_V  = R*PAD_LP;
    localparam integer B_U  = LP*R;
    localparam integer B_P  = D*D;

    reg signed [15:0] w_dw [0:2*B_DW-1];
    reg signed [15:0] w_v  [0:2*B_V-1];
    reg signed [15:0] w_u  [0:2*B_U-1];
    reg signed [15:0] w_p  [0:2*B_P-1];
    // Bias of the pointwise projection, one constant per output feature.
    reg signed [15:0] b_p  [0:2*D-1];

    // The lanes of one cycle read consecutive features, so the tables are
    // repacked once at time zero into words that hold a whole lane group. One
    // lookup then serves all W lanes, which is what keeps the ROM out of the
    // multiplier array's address logic.
    integer p0, c0, k0, j0;
    reg [16*W-1:0] w_dw_w [0:2*BLK*PAD_KL-1];
    reg [16*W-1:0] w_p_w  [0:2*BLK*D-1];
    reg [16*W-1:0] b_p_w  [0:2*BLK-1];

    initial begin
        $readmemh("w_dw_pad.hex",   w_dw, 0, B_DW-1);
        $readmemh("w_dw_pad_1.hex", w_dw, B_DW, 2*B_DW-1);
        $readmemh("w_v_pad.hex",    w_v, 0, B_V-1);
        $readmemh("w_v_pad_1.hex",  w_v, B_V, 2*B_V-1);
        $readmemh("w_u.hex",        w_u, 0, B_U-1);
        $readmemh("w_u_1.hex",      w_u, B_U, 2*B_U-1);
        $readmemh("w_p.hex",        w_p, 0, B_P-1);
        $readmemh("w_p_1.hex",      w_p, B_P, 2*B_P-1);
        $readmemh("b_p.hex",        b_p, 0, D-1);
        $readmemh("b_p_1.hex",      b_p, D, 2*D-1);
        for (p0 = 0; p0 < 2; p0 = p0 + 1) begin
            for (c0 = 0; c0 < BLK; c0 = c0 + 1) begin
                for (k0 = 0; k0 < PAD_KL; k0 = k0 + 1)
                    for (j0 = 0; j0 < W; j0 = j0 + 1)
                        w_dw_w[p0*BLK*PAD_KL + c0*PAD_KL + k0][j0*16 +: 16]
                            = w_dw[p0*B_DW + (c0*W + j0)*PAD_KL + k0];
                for (k0 = 0; k0 < D; k0 = k0 + 1)
                    for (j0 = 0; j0 < W; j0 = j0 + 1)
                        w_p_w[p0*BLK*D + c0*D + k0][j0*16 +: 16]
                            = w_p[p0*B_P + (c0*W + j0)*D + k0];
                for (j0 = 0; j0 < W; j0 = j0 + 1)
                    b_p_w[p0*BLK + c0][j0*16 +: 16] = b_p[p0*D + c0*W + j0];
            end
        end
    end

    // ---- lattice memories ------------------------------------------------
    wire [AW-1:0] ra_z, ra_h, ra_g;
    wire [WW-1:0] rd_z, rd_h, rd_g;

    reg  [AW-1:0] waddr;
    reg           we_z, we_h, we_g;
    reg  [WW-1:0] wdata;
    reg signed [15:0] wv;

    dpram #(.WIDTH(WW), .DEPTH(DEP), .AW(AW)) u_z (.clk(clk), .we(we_z),
        .waddr(waddr), .wdata(wdata), .raddr(ra_z), .rdata(rd_z));
    dpram #(.WIDTH(WW), .DEPTH(DEP), .AW(AW)) u_h (.clk(clk), .we(we_h),
        .waddr(waddr), .wdata(wdata), .raddr(ra_h), .rdata(rd_h));
    dpram #(.WIDTH(WW), .DEPTH(DEP), .AW(AW)) u_g (.clk(clk), .we(we_g),
        .waddr(waddr), .wdata(wdata), .raddr(ra_g), .rdata(rd_g));

    // ---- load ------------------------------------------------------------
    reg  [15:0] ld_cnt;
    reg  [WW-1:0] ld_buf;
    wire [WW-1:0] ld_word = (LD_SERIAL != 0) ? ld_buf : ld_data;
    integer q;
    always @(posedge clk) begin
        if (!rst_n) begin ld_cnt <= 0; ld_buf <= 0; end
        else if (ld_en) begin
            if (LD_SERIAL != 0) begin
                ld_buf[ld_cnt*16 +: 16] <= ld_data;
                if (ld_cnt == W-1) ld_cnt <= 0; else ld_cnt <= ld_cnt + 1;
            end
        end
    end

    wire ld_we = ld_en && ((LD_SERIAL == 0) || (ld_cnt == W-1));

    // ---- counters --------------------------------------------------------
    reg  [3:0]  st;
    reg  [7:0]  pass;
    reg  [2:0]  prime;
    reg  [15:0] a, c, k;
    reg  [15:0] a1, c1, k1;
    reg  [3:0]  st1;
    reg  [15:0] a2, c2;
    reg  [3:0]  st2;
    // The bank index has to travel with the pipeline: the last write of a pass
    // lands a cycle after `pass` has already moved on.
    reg  [7:0]  pass1, pass2;

    // ---- stage 0: addresses and weight select ---------------------------
    reg  [AW-1:0] s0_z, s0_h, s0_g;
    reg  signed [15:0] wsel [0:W-1];
    integer j;
    reg signed [15:0] src;
    reg  [15:0] kk;
    reg  [16*W-1:0] wdw, wpw;

    always @(*) begin
        s0_z = rd_addr_e[AW-1:0]; s0_h = 0; s0_g = 0;
        for (j = 0; j < W; j = j + 1) wsel[j] = 16'sd0;
        case (st)
        S_DW: begin
            src = a + k - (KL/2);
            if (src < 0) src = src + LP;
            if (src >= LP) src = src - LP;
            s0_z = src * BLK + c;
            wdw = w_dw_w[pass*BLK*PAD_KL + c*PAD_KL + k];
            for (j = 0; j < W; j = j + 1)
                wsel[j] = wdw[j*16 +: 16];
        end
        S_DOWN: begin
            s0_h = k * BLK + c;
            for (j = 0; j < W; j = j + 1)
                wsel[j] = w_v[pass * B_V + a * PAD_LP + k];
        end
        S_UP: begin
            s0_g = k * BLK + c;
            s0_h = a * BLK + c;
            for (j = 0; j < W; j = j + 1)
                wsel[j] = w_u[pass * B_U + a * R + k];
        end
        S_POINT: begin
            s0_h = a * BLK + (k / W);
            s0_z = a * BLK + c;
            wpw = w_p_w[pass*BLK*D + c*D + k];
            for (j = 0; j < W; j = j + 1)
                wsel[j] = wpw[j*16 +: 16];
        end
        default: ;
        endcase
    end

    assign ra_z = s0_z;
    assign ra_h = s0_h;
    assign ra_g = s0_g;

    // Residual folded into the accumulator on the first tap of a row. The
    // residual of block c is the word at (a, c): one value per lane.

    // ---- stage 1: multiply and accumulate -------------------------------
    reg signed [43:0] acc [0:W-1];
    reg signed [15:0] wsel1 [0:W-1];
    reg signed [15:0] lane_a [0:W-1];
    // The value the current tap contributes, combinationally: the write happens
    // on the same edge the accumulator is updated, so a registered copy would
    // lag one tap behind.
    reg signed [43:0] cur [0:W-1];
    reg signed [43:0] res_scaled [0:W-1];

    function signed [15:0] sat;
        input signed [43:0] v;
        reg signed [43:0] s;
        begin
            s = (v + (44'sd1 <<< (WQ-1))) >>> WQ;
            if (s > 44'sd32767) sat = 16'sd32767;
            else if (s < -44'sd32768) sat = -16'sd32768;
            else sat = s[15:0];
        end
    endfunction

    wire [16*W-1:0] bp_word = b_p_w[pass2*BLK + c1];
    wire [WW-1:0] st1_word = (st1 == S_DW)   ? rd_z :
                             (st1 == S_DOWN) ? rd_h :
                             (st1 == S_UP)   ? rd_g : rd_h;

    integer m;
    always @(*) begin
        for (m = 0; m < W; m = m + 1) begin
            // operands first, the sum last: a blocking assignment order that
            // reads res_scaled before it is set folds in the previous tap
            //
            // S_POINT walks the input feature axis, so every lane reads the
            // same lattice element and only the weight differs. rd_h already
            // holds the word addressed in the previous cycle, which is the one
            // stage 1 wants: a registered copy would lag a feature behind.
            lane_a[m] = (st1 == S_POINT) ? rd_h[(k1 % W)*16 +: 16]
                                         : st1_word[m*16 +: 16];
            // residual read here rather than in a helper block, and indexed
            // by the loop variable: a shared temporary took its last value
            if (st1 == S_UP)
                res_scaled[m] = {{28{rd_h[m*16 + 15]}}, rd_h[m*16 +: 16]} <<< WQ;
            else if (st1 == S_POINT)
                res_scaled[m] = {{28{rd_z[m*16 + 15]}}, rd_z[m*16 +: 16]} <<< WQ;
            else
                res_scaled[m] = 44'sd0;
            cur[m] = (k1 == 0) ? (lane_a[m] * wsel1[m] + res_scaled[m])
                               : (acc[m] + lane_a[m] * wsel1[m]);
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            st1 <= S_IDLE; a1 <= 0; c1 <= 0; k1 <= 0; v1 <= 0; pass1 <= 0; pass2 <= 0;
            for (m = 0; m < W; m = m + 1) begin acc[m] <= 0; wsel1[m] <= 0; end
        end else if (st == S_IDLE) begin
            // Idle is a clean slate: the tail of the previous run must not be
            // able to issue a write when the next one starts.
            st1 <= S_IDLE; a1 <= 0; c1 <= 0; k1 <= 0; v1 <= 0;
            for (m = 0; m < W; m = m + 1) begin acc[m] <= 0; wsel1[m] <= 0; end
        end else begin
            st1 <= st; a1 <= a; c1 <= c; k1 <= k;
            for (m = 0; m < W; m = m + 1) wsel1[m] <= wsel[m];
            st2 <= st1; a2 <= a1; c2 <= c1;
            pass1 <= pass; pass2 <= pass1;

            // Only accumulate for taps stage 0 actually issued: during the drain
            // at a state boundary stage 1 still carries the previous state's
            // flags and would fold stale products in.
            v1 <= v0;
            if (v1)
                for (m = 0; m < W; m = m + 1)
                    acc[m] <= (k1 == 0) ? (lane_a[m] * wsel1[m] + res_scaled[m])
                                        : (acc[m] + lane_a[m] * wsel1[m]);
        end
    end

    // ---- stage 2: write back --------------------------------------------
    function [15:0] last_tap;
        input [3:0] s;
        begin
            case (s)
            S_DW:    last_tap = KL - 1;
            S_DOWN:  last_tap = LP - 1;
            S_UP:    last_tap = R - 1;
            S_POINT: last_tap = D - 1;
            default: last_tap = 0;
            endcase
        end
    endfunction

    // A tap is real only when stage 0 issued it with the pipeline primed.
    // Without this the preheat cycles repeat the first tap of a row, and since
    // S_UP writes h in place the repeat reads back its own output and folds the
    // up-projection in a second time.
    wire v0 = (st == S_DW || st == S_DOWN || st == S_UP || st == S_POINT)
              && (prime == 0);
    reg  v1;
    wire do_write = v1 && (k1 == last_tap(st1))
                    && (st1 == S_DW || st1 == S_DOWN || st1 == S_UP || st1 == S_POINT);

    // Each state owns one lattice region. A shared write enable would have the
    // convolution result land in z and g at the same time.
    always @(posedge clk) begin
        if (!rst_n) begin
            we_z <= 0; we_h <= 0; we_g <= 0; waddr <= 0; wdata <= 0;
        end else if (ld_en) begin
            we_z <= 1'b1; we_h <= 1'b0; we_g <= 1'b0;
            waddr <= ld_addr[AW-1:0];
            if (LD_SERIAL != 0) begin
                // the buffer holds the earlier lanes only, so the lane arriving
                // this cycle has to be patched in before the word is written
                wdata <= ld_buf;
                wdata[ld_cnt*16 +: 16] <= ld_data;
            end else wdata <= ld_data;
        end else if (do_write) begin
            we_z <= (st1 == S_POINT);
            we_h <= (st1 == S_DW) || (st1 == S_UP);
            we_g <= (st1 == S_DOWN);
            waddr <= a1 * BLK + c1;
            for (m = 0; m < W; m = m + 1) begin
                // the pointwise projection is affine: its bias joins the
                // accumulator on the last tap, before the single rounding
                if (st1 == S_POINT)
                    wv = sat(cur[m] + ($signed({{28{bp_word[m*16 + 15]}},
                                               bp_word[m*16 +: 16]}) <<< 12));
                else
                    wv = sat(cur[m]);
                // the activation sits on the up-projection result only
                if (ACT_RELU != 0 && st1 == S_UP && wv[15]) wv = 16'sd0;
                wdata[m*16 +: 16] <= wv;
            end
        end else begin
            we_z <= 1'b0; we_h <= 1'b0; we_g <= 1'b0;
        end
    end

    assign dbg = rd_z[0];
    assign rd_word = rd_z;

    // ---- sequencing ------------------------------------------------------
    wire [15:0] a_last = (st == S_DOWN) ? R-1 : LP-1;
    wire at_end = (k == last_tap(st)) && (c == BLK-1) && (a == a_last);

    reg [15:0] nxt_a, nxt_c, nxt_k;
    always @(*) begin
        nxt_a = a; nxt_c = c; nxt_k = k;
        if (k != last_tap(st)) nxt_k = k + 1;
        else if (c != BLK-1) begin nxt_k = 0; nxt_c = c + 1; end
        else begin
            nxt_k = 0; nxt_c = 0;
            nxt_a = (a == a_last) ? 0 : a + 1;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 0; prime <= 0; pass <= 0;
            a <= 0; c <= 0; k <= 0;
        end else begin
            case (st)
            // done has to drop when a new run is launched, otherwise a caller
            // that watches it cannot tell the second run from the first
            // done stays high after a run so a caller can sample it late, and
            // drops when the next run is launched
            S_IDLE: if (start) begin
                done <= 1'b0; pass <= 0;
                a <= 0; c <= 0; k <= 0; st <= S_DW; prime <= 3'd4;
            end
            S_DW, S_DOWN, S_UP, S_POINT: begin
                // the first taps of a state arrive before the pipeline has
                // filled, so hold the counters until the stages line up
                if (prime != 0) prime <= prime - 1;
                else if (at_end) begin
                    a <= 0; c <= 0; k <= 0;
                    if (st == S_DW)         begin st <= S_DOWN;  prime <= 3'd4; end
                    else if (st == S_DOWN)  begin st <= S_UP;    prime <= 3'd4; end
                    else if (st == S_UP)    begin st <= S_POINT; prime <= 3'd4; end
                    else if (pass + 1 < NPASS) begin
                        // run the same block again on its own output
                        pass <= pass + 1; st <= S_DW; prime <= 3'd4;
                    end
                    else begin st <= S_DONE; done <= 1'b1; end
                end else begin
                    a <= nxt_a; c <= nxt_c; k <= nxt_k;
                end
            end
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
