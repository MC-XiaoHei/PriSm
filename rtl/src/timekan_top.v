// TimeKAN, whole model, one variable at a time.
//
// Lookback 96, prediction 96, two levels (96 and 48), 32 channels, one encoder
// layer, Chebyshev degrees 0 and 1, channel independent.
//
// Data is Q12, weights Q14, one truncating shift at the end of each
// accumulation. Memory reads are registered, so each dot product spends one
// extra cycle filling the pipeline.
//
// The bank map is fixed up front because the whole model has to live in block
// RAM: 14,112 words of scratch and 18,752 words of weights.

`timescale 1ns / 1ps

module timekan_top (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire                 ld_en,
    input  wire [1:0]           ld_sel,     // 0 weights, 1 tables, 2 input
    input  wire [15:0]          ld_addr,
    input  wire signed [15:0]   ld_data,
    output reg                  done,
    output wire [15:0]          rd_addr,
    output wire signed [15:0]   rd_data
);

    localparam integer L  = 96;
    localparam integer L1 = 48;
    localparam integer DM = 32;

    localparam integer X_X   = 0;
    localparam integer X_N0  = 96;
    localparam integer X_L1  = 192;
    localparam integer X_N1  = 240;
    localparam integer XMEM  = 288;
    localparam integer I_E0  = 288;
    localparam integer I_E1  = 3360;
    localparam integer I_HI  = 4896;
    localparam integer I_TMP = 7968;
    localparam integer I_LOW = 11040;
    localparam integer I_H2  = 12576;
    localparam integer I_MIX = 15648;
    localparam integer AMEM  = 18720;

    localparam integer W_EMB   = 0;
    localparam integer W_KANL  = 96;
    localparam integer W_KANH  = 1120;
    localparam integer W_CONVL = 3168;
    localparam integer W_CONVH = 3264;
    localparam integer W_PRED  = 3360;
    localparam integer W_PREDB = 12576;
    localparam integer W_PROJ  = 12672;
    localparam integer W_PROJB = 12704;
    localparam integer W_TANH  = 12736;
    localparam integer W_ACOS  = 12992;
    localparam integer W_COS   = 13248;
    localparam integer W_INTR  = 14272;
    localparam integer WMEM    = 18880;

    localparam integer WQ = 14;
    localparam integer DQ = 12;

    localparam [4:0] S_IDLE = 0,
                     S_SUM0 = 1, S_SQ0 = 2, S_SQRT0 = 3, S_INV0 = 4, S_APP0 = 5,
                     S_DN1 = 6, S_SUM1 = 7, S_SQ1 = 8, S_SQRT1 = 9, S_INV1 = 10, S_APP1 = 11,
                     S_EMB0 = 12, S_EMB1 = 13,
                     S_UP = 14, S_HI = 15,
                     S_MIXL = 16, S_CONVL = 17,
                     S_BASIS = 18, S_MIXH = 27, S_CONVH = 19,
                     S_UP2 = 20, S_MIX2 = 21,
                     S_DENSE = 22, S_PROJ = 23, S_DENORM = 24, S_EMIT = 25, S_DONE = 26,
                     S_BIAS = 28;

    reg  [4:0]  st;
    reg  [15:0] t, o, i;
    reg  [1:0]  ph;
    reg  signed [63:0] acc;
    reg  signed [63:0] sum65;
    reg  signed [31:0] mean_r, var_r, inv_r;
    reg  [15:0] ra, rb, rb2;
    reg  [15:0] tbl_addr;
    reg  signed [15:0] q_a, q_b, q_b2;
    reg  signed [15:0] kan_hold;
    reg  [15:0] hi_t, hi_o;
    reg         hi_run, hi_last;
    reg  [15:0] out_addr;
    reg  signed [15:0] outv;
    reg         out_we;
    reg  [15:0] ld_base;
    reg signed [15:0] lvl0_mean, lvl0_std;
    reg  signed [15:0] bx0 [0:31];
    reg  signed [15:0] bx1 [0:31];
    reg  [15:0] out_word;

    reg  [31:0] dv_arg, dv_div, dv_num, dv_den, dv_q, dv_r;
    reg  [5:0]  dv_cnt;
    reg         dv_go, dv_busy, dv_done;
    reg  [39:0] sq_x, sq_res, sq_bit, sq_arg;
    reg         sq_go, sq_busy, sq_done;

    // two copies, one per read port: a single-port block RAM cannot serve
    // the two simultaneous reads this model needs, and duplicating the
    // storage is cheaper than falling back to distributed RAM
    (* ram_style = "block", syn_ramstyle = "block_ram" *) reg signed [15:0] bmem  [0:AMEM-1];
    (* ram_style = "block", syn_ramstyle = "block_ram" *) reg signed [15:0] bmem2 [0:AMEM-1];
    (* syn_ramstyle = "distributed_ram" *) reg signed [15:0] xmem [0:XMEM-1];
    (* ram_style = "block", syn_ramstyle = "block_ram" *) reg signed [15:0] wmem [0:WMEM-1];

    // The activation store has one write port. Each stage that stores a row
    // fills the port and the write itself happens once, at the end of the step.
    reg                bw_en;
    reg  [15:0]        bw_addr;
    reg  signed [15:0] bw_data;

    // A ring index for the circular padding. The argument is always below
    // three times the modulus, so two conditional subtractions replace the
    // divider the modulo operator would build.
    function [15:0] wrap3;
        input [15:0] v;
        input [15:0] m;
        begin
            if (v >= m + m) wrap3 = v - m - m;
            else if (v >= m) wrap3 = v - m;
            else wrap3 = v;
        end
    endfunction

    always @(posedge clk) begin
        q_a  <= bmem[ra];
        q_b2 <= bmem2[rb2];
        q_b  <= wmem[rb];
    end

    always @(posedge clk) begin
        if (ld_en) begin
            if (ld_sel == 2'd2) xmem[ld_addr] <= ld_data;
            else wmem[ld_addr] <= ld_data;
        end
    end

    // ---- restoring divider ----------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            dv_q <= 0; dv_r <= 0; dv_num <= 0; dv_cnt <= 0; dv_busy <= 0; dv_done <= 0;
        end else begin
            dv_done <= 1'b0;
            if (dv_go) begin
                dv_num <= dv_arg; dv_den <= (dv_div == 0) ? 32'd1 : dv_div;
                dv_q <= 0; dv_r <= 0; dv_cnt <= 0; dv_busy <= 1'b1;
            end else if (dv_busy) begin
                if ({dv_r[30:0], dv_num[31]} >= dv_den) begin
                    dv_r <= {dv_r[30:0], dv_num[31]} - dv_den;
                    dv_q <= {dv_q[30:0], 1'b1};
                end else begin
                    dv_r <= {dv_r[30:0], dv_num[31]};
                    dv_q <= {dv_q[30:0], 1'b0};
                end
                dv_num <= {dv_num[30:0], 1'b0};
                dv_cnt <= dv_cnt + 1;
                if (dv_cnt == 6'd31) begin dv_busy <= 1'b0; dv_done <= 1'b1; end
            end
        end
    end

    // ---- square root -----------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            sq_x <= 0; sq_res <= 0; sq_bit <= 0; sq_busy <= 0; sq_done <= 0;
        end else begin
            sq_done <= 1'b0;
            if (sq_go) begin
                sq_x <= sq_arg; sq_res <= 0; sq_bit <= 40'h40_0000_0000; sq_busy <= 1'b1;
            end else if (sq_busy) begin
                if (sq_bit == 0) begin sq_busy <= 1'b0; sq_done <= 1'b1; end
                else begin
                    if (sq_x >= sq_res + sq_bit) begin
                        sq_x   <= sq_x - (sq_res + sq_bit);
                        sq_res <= (sq_res >> 1) + sq_bit;
                    end else sq_res <= (sq_res >> 1);
                    sq_bit <= (sq_bit == 1) ? 40'd0 : (sq_bit >> 2);
                end
            end
        end
    end

    function signed [15:0] clamp_shift;
        input signed [63:0] v;
        input integer sh;
        reg signed [63:0] s;
        begin
            s = v >>> sh;
            if (s > 32767) clamp_shift = 16'sd32767;
            else if (s < -32768) clamp_shift = -16'sd32768;
            else clamp_shift = s[15:0];
        end
    endfunction

    function signed [15:0] clip16;
        input signed [63:0] v;
        begin
            if (v > 32767) clip16 = 16'sd32767;
            else if (v < -32768) clip16 = -16'sd32768;
            else clip16 = v[15:0];
        end
    endfunction

    // ---- address generation ----------------------------------------------
    always @(*) begin
        ra = 0; rb = 0;
        case (st)
        S_UP:    ra = I_E1 + i * DM + o;
        S_UP2:   ra = I_LOW + i * DM + o;

        S_HI:    ra = I_E0 + t * DM + o;
        S_BASIS: ra = I_HI + t * DM + i;
        S_MIXL:  ra = I_E1 + t * DM + i;
        S_MIXH:  ra = I_HI + t * DM + i;
        S_CONVL: ra = I_E1 + wrap3(t + i + L1 - 1, L1) * DM + o;
        S_CONVH: ra = I_HI + wrap3(t + i + L - 1, L) * DM + o;
        S_DENSE: ra = I_MIX + i * DM + o;
        S_PROJ:  ra = I_TMP + t * DM + i;
        default: ;
        endcase
        rb2 = 0;
        case (st)
        S_HI:    rb2 = I_TMP + t * DM + o;
        S_CONVL, S_CONVH: rb2 = I_TMP + t * DM + o;
        S_UP2:   rb2 = I_H2 + t * DM + o;
        default: ;
        endcase
        case (st)
        S_BASIS: rb = tbl_addr;
        S_EMB0, S_EMB1: rb = W_EMB + o * 3 + i;
        S_UP, S_UP2:    rb = W_INTR + t * L1 + i;
        S_MIXH:         rb = W_KANH + ((ph == 2) ? (i + 16'd1) : i) * DM * 2 + o * 2
                              + ((ph == 1) ? 16'd1 : 16'd0);
        S_MIXL:         rb = W_KANL + i * DM + o;
        S_CONVL:        rb = W_CONVL + o * 3 + i;
        S_CONVH:        rb = W_CONVH + o * 3 + i;
        S_DENSE:        rb = W_PRED + t * L + i;
        S_PROJ:         rb = W_PROJ + i;
        S_BIAS:  rb = (i < L) ? (W_PREDB + i) : W_PROJB;
        default: ;
        endcase
    end

    wire signed [63:0] prod  = q_a * q_b;
    wire signed [63:0] acc_n = acc + prod;
    // the convolution pads with zeros, not by wrapping: a tap outside the band
    // contributes nothing. The value in hand corresponds to the index issued
    // one cycle earlier, hence the shift by two.
    wire tap_l = (t + i >= 2) && ((t + i) < L1 + 2);
    wire tap_h = (t + i >= 2) && ((t + i) < L + 2);
    wire signed [63:0] acc_l = acc + (tap_l ? prod : 64'sd0);
    // degree-0 basis: the multiplicand is the constant one, not the activation
    wire signed [63:0] acc_k = acc + (16'sd4096 * q_b);
    wire signed [63:0] acc_h = acc + (tap_h ? prod : 64'sd0);

    // the three tables are addressed through `tbl_addr`, which is registered,
    // so the table bank stays in block RAM
    reg signed [15:0] pbias [0:95];
    reg signed [15:0] projbias;
    reg  [3:0]  bp;
    wire signed [15:0] tanh_idx_of = {4'b0, (((q_b + 16'sd32768) >>> 8) & 16'h00FF)};
    wire [15:0] acos_idx_of = ((q_b + 16'sd4096) >>> 5) & 16'h00FF;
    wire [15:0] cos_idx_of  = ((($signed(q_b) * 32'sd163) >>> 12)) & 16'h03FF;

    assign rd_addr = out_addr;
    assign rd_data = outv;

    // each dot product: one cycle to issue the first address, then one cycle
    // per tap, then one to write back
    wire dot_start = (ph == 0);
    wire signed [15:0] qq = clamp_shift(acc_n, WQ);

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 0; t <= 0; o <= 0; i <= 0;
            ph <= 0; acc <= 0; sum65 <= 0; mean_r <= 0; var_r <= 0; inv_r <= 0;
            out_we <= 0; out_addr <= 0; outv <= 0; dv_go <= 0; sq_go <= 0; hi_run <= 0; hi_last <= 0; bp <= 0;
            lvl0_mean <= 0; lvl0_std <= 0; ld_base <= 0;
        end else begin
            out_we <= 1'b0; dv_go <= 1'b0; sq_go <= 1'b0;

            case (st)
            S_IDLE: if (start) begin
                st <= S_BIAS; i <= 0; t <= 0; o <= 0; ph <= 0; sum65 <= 0;
                done <= 1'b0;
            end

            // the bias vectors leave the weight bank once, into registers, so
            // that bank keeps a single registered read port. The value visible
            // in a cycle belongs to the address issued in the previous one.
            S_BIAS: begin
                if (i >= 1 && i <= L) pbias[i - 1] <= q_b;
                if (i == L + 1) begin projbias <= q_b; i <= 0; st <= S_SUM0; end
                else i <= i + 1;
            end

            // ---- level 0 statistics --------------------------------------
            S_SUM0: begin
                if (i < L) begin sum65 <= sum65 + $signed(xmem[X_X + i]); i <= i + 1; end
                else begin
                    mean_r <= sum65 / L;
                    lvl0_mean <= (sum65 / L);
                    sum65 <= 0; i <= 0; st <= S_SQ0;
                end
            end
            S_SQ0: begin
                if (i < L) begin
                    sum65 <= sum65 + (($signed(xmem[X_X + i]) - mean_r) * ($signed(xmem[X_X + i]) - mean_r));
                    i <= i + 1;
                end else begin
                    var_r <= sum65 / L;
                    sq_arg <= (sum65 / L > 0) ? sum65[39:0] / L : 40'd0;
                    sum65 <= 0; i <= 0; sq_go <= 1'b1; st <= S_SQRT0;
                end
            end
            S_SQRT0: if (sq_done) begin
                lvl0_std <= sq_res[15:0];
                dv_arg <= 32'd16777216; dv_div <= (sq_res == 0) ? 32'd1 : sq_res[31:0];
                dv_go <= 1'b1; st <= S_INV0;
            end
            S_INV0: if (dv_done) begin inv_r <= dv_q; i <= 0; st <= S_APP0; end
            S_APP0: begin
                xmem[X_N0 + i] <= clamp_shift((($signed(xmem[X_X + i]) - mean_r) * inv_r), DQ);
                if (i == L - 1) begin i <= 0; t <= 0; st <= S_DN1; end
                else i <= i + 1;
            end

            // ---- level 1 --------------------------------------------------
            S_DN1: begin
                xmem[X_L1 + t] <= (($signed(xmem[X_X + 2 * t]) + $signed(xmem[X_X + 2 * t + 1])) >>> 1);
                if (t == L1 - 1) begin t <= 0; i <= 0; sum65 <= 0; st <= S_SUM1; end
                else t <= t + 1;
            end
            S_SUM1: begin
                if (i < L1) begin sum65 <= sum65 + $signed(xmem[X_L1 + i]); i <= i + 1; end
                else begin mean_r <= sum65 / L1; sum65 <= 0; i <= 0; st <= S_SQ1; end
            end
            S_SQ1: begin
                if (i < L1) begin
                    sum65 <= sum65 + (($signed(xmem[X_L1 + i]) - mean_r) * ($signed(xmem[X_L1 + i]) - mean_r));
                    i <= i + 1;
                end else begin
                    sq_arg <= (sum65 / L1 > 0) ? sum65[39:0] / L1 : 40'd0;
                    sum65 <= 0; i <= 0; sq_go <= 1'b1; st <= S_SQRT1;
                end
            end
            S_SQRT1: if (sq_done) begin
                dv_arg <= 32'd16777216; dv_div <= (sq_res == 0) ? 32'd1 : sq_res[31:0];
                dv_go <= 1'b1; st <= S_INV1;
            end
            S_INV1: if (dv_done) begin inv_r <= dv_q; i <= 0; st <= S_APP1; end
            S_APP1: begin
                xmem[X_N1 + i] <= clamp_shift((($signed(xmem[X_L1 + i]) - mean_r) * inv_r), DQ);
                if (i == L1 - 1) begin i <= 0; t <= 0; o <= 0; ph <= 0; acc <= 0; st <= S_EMB0; end
                else i <= i + 1;
            end

            // ---- embedding, circular padding ------------------------------
            S_EMB0: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == 3) begin
                    bw_en = 1'b1; bw_addr = I_E0 + t * DM + o; bw_data = clamp_shift(acc + (xmem[X_N0 + wrap3(t + i + L - 2, L)] * q_b), WQ);
                    ph <= 0; i <= 0; acc <= 0;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L - 1) begin t <= 0; st <= S_EMB1; end else t <= t + 1;
                    end else o <= o + 1;
                end else begin
                    acc <= acc + (xmem[X_N0 + wrap3(t + i + L - 2, L)] * q_b);
                    i <= i + 1;
                end
            end
            S_EMB1: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == 3) begin
                    bw_en = 1'b1; bw_addr = I_E1 + t * DM + o; bw_data = clamp_shift(acc + (xmem[X_N1 + wrap3(t + i + L1 - 2, L1)] * q_b), WQ);
                    ph <= 0; i <= 0; acc <= 0;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L1 - 1) begin t <= 0; o <= 0; st <= S_UP; end else t <= t + 1;
                    end else o <= o + 1;
                end else begin
                    acc <= acc + (xmem[X_N1 + wrap3(t + i + L1 - 2, L1)] * q_b);
                    i <= i + 1;
                end
            end

            // ---- upsample the low band ------------------------------------
            S_UP: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == L1) begin
                    bw_en = 1'b1; bw_addr = I_TMP + t * DM + o; bw_data = qq;
                    ph <= 0; i <= 0; acc <= 0;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L - 1) begin t <= 0; o <= 0; st <= S_HI; end else t <= t + 1;
                    end else o <= o + 1;
                end else begin acc <= acc_n; i <= i + 1; end
            end
            S_HI: begin
                // both operands arrive one cycle after their addresses, so the
                // write lags the counters by one, and the stage needs one more
                // cycle at the end to let the last write land
                if (hi_run) begin bw_en = 1'b1; bw_addr = I_HI + hi_t * DM + hi_o; bw_data = clamp_shift($signed(q_a) - $signed(q_b2), 0); end
                hi_t <= t; hi_o <= o;
                if (hi_last) begin
                    hi_last <= 1'b0; hi_run <= 1'b0;
                    t <= 0; o <= 0; i <= 0; acc <= 0; ph <= 0; st <= S_MIXL;
                end else begin
                    hi_run <= 1'b1;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L - 1) hi_last <= 1'b1; else t <= t + 1;
                    end else o <= o + 1;
                end
            end

            // ---- low band: degree-0 KAN is a constant vector --------------
            S_MIXL: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == DM) begin
                    bw_en = 1'b1; bw_addr = I_TMP + t * DM + o; bw_data = clamp_shift(acc_k, WQ);
                    ph <= 0; i <= 0; acc <= 0;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L1 - 1) begin t <= 0; o <= 0; st <= S_CONVL; end else t <= t + 1;
                    end else o <= o + 1;
                end else begin acc <= acc_k; i <= i + 1; end
            end
            S_CONVL: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == 1) begin kan_hold <= q_b2; acc <= acc_l; i <= i + 1; end
                else if (i == 3) begin
                    bw_en = 1'b1; bw_addr = I_LOW + t * DM + o; bw_data = clamp_shift(acc_l + (($signed(kan_hold) <<< WQ)), WQ);
                    ph <= 0; i <= 0; acc <= 0;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L1 - 1) begin t <= 0; o <= 0; i <= 0; ph <= 0; st <= S_BASIS; end else t <= t + 1;
                    end else o <= o + 1;
                end else begin acc <= acc_l; i <= i + 1; end
            end

            // ---- high band basis, one input per two cycles ----------------
            S_BASIS: begin
                // x -> tanh -> tanh -> acos -> cos. A table value appears two
                // cycles after its address is set, so the chain runs on even
                // cycles and the odd ones just wait.
                case (bp)
                4'd0: bp <= 4'd1;
                4'd1: begin tbl_addr <= W_TANH + (((q_a + 16'sd32768) >>> 8) & 16'h00FF); bp <= 4'd2; end
                4'd2: bp <= 4'd3;
                4'd3: begin tbl_addr <= W_TANH + tanh_idx_of[15:0]; bp <= 4'd4; end
                4'd4: bp <= 4'd5;
                4'd5: begin tbl_addr <= W_ACOS + acos_idx_of; bp <= 4'd6; end
                4'd6: bp <= 4'd7;
                4'd7: begin tbl_addr <= W_COS + cos_idx_of; bp <= 4'd8; end
                4'd8: bp <= 4'd9;
                default: begin
                    bx0[i] <= 16'sd4096;
                    bx1[i] <= q_b;
                    bp <= 4'd0;
                    if (i == DM - 1) begin i <= 0; acc <= 0; st <= S_MIXH; end
                    else i <= i + 1;
                end
                endcase
            end

            // ---- high band: degree-1 KAN, two coefficients per input -------
            S_MIXH: begin
                if (ph == 0) begin ph <= 1; i <= 0; acc <= 0; end
                else if (ph == 1) begin
                    // q_b holds c0 of input i; c1 was just issued
                    acc <= acc + (bx0[i] * q_b);
                    ph <= 2;
                end else begin
                    acc <= acc + (bx1[i] * q_b);
                    ph <= 1;
                    if (i == DM - 1) begin
                        bw_en = 1'b1; bw_addr = I_TMP + t * DM + o; bw_data = clamp_shift(acc + (bx1[i] * q_b), WQ);
                        i <= 0; acc <= 0; ph <= 0;
                        if (o == DM - 1) begin
                            o <= 0;
                            if (t == L - 1) begin t <= 0; o <= 0; i <= 0; ph <= 0; acc <= 0; st <= S_CONVH; end
                            else begin t <= t + 1; i <= 0; st <= S_BASIS; end
                        end else o <= o + 1;
                    end else i <= i + 1;
                end
            end
            S_CONVH: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == 1) begin kan_hold <= q_b2; acc <= acc_h; i <= i + 1; end
                else if (i == 3) begin
                    bw_en = 1'b1; bw_addr = I_H2 + t * DM + o; bw_data = clamp_shift(acc_h + (($signed(kan_hold) <<< WQ)), WQ);
                    ph <= 0; i <= 0; acc <= 0;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L - 1) begin t <= 0; o <= 0; st <= S_UP2; end else t <= t + 1;
                    end else o <= o + 1;
                end else begin acc <= acc_h; i <= i + 1; end
            end

            // ---- add the upsampled low band back --------------------------
            S_UP2: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == 1) begin kan_hold <= q_b2; acc <= acc_n; i <= i + 1; end
                else if (i == L1) begin
                    bw_en = 1'b1; bw_addr = I_MIX + t * DM + o; bw_data = clamp_shift(($signed(kan_hold) <<< WQ) + acc_n, WQ);
                    ph <= 0; i <= 0; acc <= 0;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L - 1) begin t <= 0; o <= 0; st <= S_DENSE; end else t <= t + 1;
                    end else o <= o + 1;
                end else begin acc <= acc_n; i <= i + 1; end
            end

            // ---- dense 96 -> 96 over time ---------------------------------
            S_DENSE: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == L) begin
                    bw_en = 1'b1; bw_addr = I_TMP + t * DM + o; bw_data = clip16((acc_n >>> WQ) + ($signed(pbias[t]) >>> (WQ - DQ)));
                    ph <= 0; i <= 0; acc <= 0;
                    if (o == DM - 1) begin
                        o <= 0;
                        if (t == L - 1) begin t <= 0; o <= 0; st <= S_PROJ; end else t <= t + 1;
                    end else o <= o + 1;
                end else begin acc <= acc_n; i <= i + 1; end
            end

            // ---- 32 channels to one --------------------------------------
            S_PROJ: begin
                if (ph == 0) begin ph <= 1; i <= 1; acc <= 0; end
                else if (i == DM) begin
                    out_word <= clip16((acc_n >>> WQ) + ($signed(projbias) >>> (WQ - DQ)));
                    ph <= 0; i <= 0; acc <= 0; st <= S_DENORM;
                end else begin acc <= acc_n; i <= i + 1; end
            end

            // undo the level-0 normalisation
            S_DENORM: begin
                outv <= clip16((($signed(out_word) * $signed(lvl0_std)) >>> DQ) + $signed(lvl0_mean));
                out_addr <= t;
                out_we <= 1'b1;
                if (t == L - 1) begin t <= 0; st <= S_DONE; done <= 1'b1; end
                else begin t <= t + 1; i <= 0; ph <= 0; acc <= 0; st <= S_PROJ; end
            end

            S_DONE: ;
            default: ;
            endcase

            if (bw_en) begin
                bmem [bw_addr] <= bw_data;
                bmem2[bw_addr] <= bw_data;
            end
        end
    end
endmodule
