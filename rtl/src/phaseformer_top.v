// PhaseFormer in fixed point.
//
// The activation store is split into one array per bank so that every bank has
// a single write port and a single read port, which is what a block RAM offers.
// The read data is registered, so each loop presents the address for the next
// element while it consumes the one that arrived this cycle: the counter runs
// one ahead of the index being accumulated, and the first cycle of every loop
// only presents an address.
`timescale 1ns / 1ps

module phaseformer_top #(
    parameter integer L    = 96,
    parameter integer P    = 24,
    parameter integer HP   = 4,
    parameter integer DM   = 32,
    parameter integer NH   = 4,
    parameter integer DH   = 8,
    parameter integer NR   = 4,
    parameter integer DFF  = 32,
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
    // bank identifiers used by the operand fetch
    localparam integer B_TOK = 0, B_Q  = 1, B_K   = 2, B_V   = 3,
                       B_CTX = 4, B_H2 = 5, B_XF  = 6, B_PR  = 7,
                       B_RCT = 8, B_RIN = 9;

    localparam integer W_EMB  = 0;
    localparam integer W_Q    = 128;
    localparam integer W_K    = 1152;
    localparam integer W_V    = 2176;
    localparam integer W_O    = 3200;
    localparam integer W_F1   = 4224;
    localparam integer W_F2   = 5248;
    localparam integer W_HEAD = 6272;
    localparam integer W_ROUT = 6400;
    localparam integer WMEM   = 6528;

    localparam [4:0] S_IDLE = 0, S_FOLD = 1, S_EMB = 2, S_Q = 3, S_K = 4,
                     S_V = 5, S_RSC = 6, S_RSM = 7, S_RCT = 8, S_RIN = 9,
                     S_RPJ = 10, S_TS = 11, S_TSM = 12, S_TCT = 13,
                     S_OPJ = 14, S_H1 = 15, S_FF1 = 16, S_GELU = 17,
                     S_FCP = 18, S_FF2 = 19, S_H2A = 20, S_HEAD = 21,
                     S_LNS = 22, S_LNM = 23, S_LNX = 24, S_LNQ = 25,
                     S_LNI = 26, S_LNA = 27, S_LNV = 28, S_EMIT = 29, S_DONE = 30;

    reg  [4:0]  st;
    reg  [15:0] t, o, i, hh, rt;

    // one array per bank, each 768 words
    reg signed [15:0] b_tok [0:767];
    reg signed [15:0] b_q   [0:767];
    reg signed [15:0] b_k   [0:767];
    reg signed [15:0] b_v   [0:767];
    reg signed [15:0] b_ctx [0:767];
    reg signed [15:0] b_h2  [0:767];

    // the small banks stay in registers; a register file reads in the same
    // cycle, so they only cost a mux tree
    reg signed [15:0] s_xi  [0:95];
    reg signed [15:0] s_xf  [0:95];
    reg signed [15:0] s_sc  [0:31];
    reg signed [15:0] s_pr  [0:31];
    reg signed [15:0] s_rct [0:127];
    reg signed [15:0] s_rin [0:31];

    reg signed [15:0] wmem [0:WMEM-1];
    reg signed [15:0] etab [0:255];
    reg signed [15:0] gtab [0:255];

    initial begin
        $readmemh("pf_exp.hex", etab);
        $readmemh("pf_gelu.hex", gtab);
        $readmemh("pf_emb.hex",  wmem, W_EMB,  W_EMB  + 128  - 1);
        $readmemh("pf_q.hex",    wmem, W_Q,    W_Q    + 1024 - 1);
        $readmemh("pf_k.hex",    wmem, W_K,    W_K    + 1024 - 1);
        $readmemh("pf_v.hex",    wmem, W_V,    W_V    + 1024 - 1);
        $readmemh("pf_o.hex",    wmem, W_O,    W_O    + 1024 - 1);
        $readmemh("pf_f1.hex",   wmem, W_F1,   W_F1   + 1024 - 1);
        $readmemh("pf_f2.hex",   wmem, W_F2,   W_F2   + 1024 - 1);
        $readmemh("pf_head.hex", wmem, W_HEAD, W_HEAD + 128  - 1);
        $readmemh("pf_rout.hex", wmem, W_ROUT, W_ROUT + 128  - 1);
    end

    // ---- restoring divider ----------------------------------------------
    // 40 bits: the variance sum for a loud token does not fit in 32
    reg  [39:0] dv_arg, dv_div, dv_n, dv_d, dv_q, dv_r;
    reg  [5:0]  dv_bit;
    reg         dv_go, dv_done, dv_neg, dv_run;

    always @(posedge clk) begin
        if (!rst_n) begin
            dv_q <= 0; dv_r <= 0; dv_done <= 0; dv_run <= 0; dv_bit <= 6'd40;
        end else begin
            dv_done <= 1'b0;
            if (dv_go) begin
                dv_n <= dv_arg; dv_d <= dv_div;
                dv_q <= 0; dv_r <= 0; dv_bit <= 0; dv_run <= 1'b1;
            end else if (dv_bit != 6'd40) begin
                dv_bit <= dv_bit + 1;
                if ({dv_r[38:0], dv_n[39]} >= dv_d) begin
                    dv_r <= {dv_r[38:0], dv_n[39]} - dv_d;
                    dv_n <= {dv_n[38:0], 1'b0};
                    dv_q <= {dv_q[38:0], 1'b1};
                end else begin
                    dv_r <= {dv_r[38:0], dv_n[39]};
                    dv_n <= {dv_n[38:0], 1'b0};
                    dv_q <= {dv_q[38:0], 1'b0};
                end
            end else if (dv_run) begin
                dv_done <= 1'b1; dv_run <= 1'b0;
            end
        end
    end

    // ---- restoring square root ------------------------------------------
    // the variance itself needs more than 32 bits before the root is taken
    reg  [39:0] sq_arg, sq_x, sq_res, sq_bit;
    reg         sq_go, sq_done;

    always @(posedge clk) begin
        if (!rst_n) begin
            sq_done <= 0; sq_bit <= 0; sq_res <= 0; sq_x <= 0;
        end else begin
            sq_done <= 1'b0;
            if (sq_go) begin
                sq_x <= sq_arg; sq_res <= 0; sq_bit <= 40'h40_0000_0000;
            end else if (sq_bit != 0) begin
                if (sq_x >= sq_res + sq_bit) begin
                    sq_x   <= sq_x - (sq_res + sq_bit);
                    sq_res <= (sq_res >> 1) + sq_bit;
                end else sq_res <= (sq_res >> 1);
                sq_bit <= sq_bit >> 2;
                if (sq_bit == 40'd1) sq_done <= 1'b1;
            end
        end
    end

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

    // 16x16 product shifted down: written inline the destination width squeezes
    // the product before the shift
    function signed [15:0] mulshr32;
        input signed [15:0] a;
        input signed [31:0] b;
        reg signed [47:0] p;
        begin
            p = a * b;
            p = p >>> 12;
            if (p > 48'sd32767) mulshr32 = 16'sd32767;
            else if (p < -48'sd32768) mulshr32 = -16'sd32768;
            else mulshr32 = p[15:0];
        end
    endfunction

    // tables are indexed inside the function so each pass can point at its own
    // score without a shared temporary
    function signed [15:0] exp_of;
        input signed [15:0] v;
        reg [15:0] idx;
        begin
            idx = (v >= 0) ? 16'd0 : (((-v) >> 5) < 16'd256 ? ((-v) >> 5) : 16'd255);
            exp_of = etab[idx];
        end
    endfunction

    // ---- operand fetch ----------------------------------------------------
    // s_bank/e_ast/e_soff describe the first operand, b_kind the second, and
    // d_bank the destination. Every read goes through a registered port, so the
    // counter i addresses the element that the next cycle will consume, and the
    // index being accumulated is j = i - 1.
    reg  [3:0]  s_bank, d_bank;
    reg  [1:0]  b_kind;              // 0 weight table, 1 value bank, 2 router context
    reg  [15:0] e_nin, e_nout, e_ast, e_soff, e_wb, e_woff, e_dast;

    reg  [9:0]  ra_tok, ra_q, ra_k, ra_v, ra_ctx, ra_h2, ra_rct, ra_pr, ra_rin, ra_xf;
    reg  [12:0] ra_w;
    reg  [15:0] lim;
    reg signed [43:0] e_acc;
    reg  [3:0] ln_bank;

    reg signed [15:0] rd_tok, rd_q, rd_k, rd_v, rd_ctx, rd_h2;
    reg signed [15:0] rd_rct, rd_pr, rd_rin, rd_w, rd_xf;

    always @(posedge clk) begin
        rd_tok <= b_tok[ra_tok];
        rd_q   <= b_q  [ra_q];
        rd_k   <= b_k  [ra_k];
        rd_v   <= b_v  [ra_v];
        rd_ctx <= b_ctx[ra_ctx];
        rd_h2  <= b_h2 [ra_h2];
        rd_rct <= s_rct[ra_rct[6:0]];
        rd_pr  <= s_pr [ra_pr[4:0]];
        rd_rin <= s_rin[ra_rin[4:0]];
        rd_xf  <= s_xf [ra_xf[6:0]];
        rd_w   <= wmem [ra_w];
    end

    wire [15:0] j  = (i == 0) ? 16'd0 : (i - 1);
    wire [15:0] ia = (i >= lim) ? (lim - 1) : i;
    wire [15:0] iw = t * e_ast + ia + e_soff;
    wire [15:0] bw = e_wb + o * e_nin + ia + e_woff;

    always @(*) begin
        ra_tok = 10'd0; ra_q = 10'd0; ra_k = 10'd0; ra_v = 10'd0;
        ra_ctx = 10'd0; ra_h2 = 10'd0; ra_rct = 10'd0; ra_pr = 10'd0;
        ra_rin = 10'd0; ra_xf = 10'd0; ra_w = 13'd0; lim = 16'd1;
        case (st)
        S_EMB, S_Q, S_K, S_V, S_RPJ, S_OPJ, S_FF1, S_FF2, S_HEAD: begin
            lim = e_nin;
            case (s_bank)
            B_TOK: ra_tok = iw[9:0];
            B_Q:   ra_q   = iw[9:0];
            B_K:   ra_k   = iw[9:0];
            B_CTX: ra_ctx = iw[9:0];
            B_H2:  ra_h2  = iw[9:0];
            B_PR:  ra_pr  = iw[6:0];
            B_RIN: ra_rin = iw[4:0];
            B_XF:  ra_xf  = iw[6:0];
            default: ;
            endcase
            case (b_kind)
            2'd0: ra_w = bw[12:0];
            2'd1: ra_v = hh * DH + i * DM + o;
            default: ra_rct = rt * DM + hh * DH + i;
            endcase
        end
        // the router weights are addressed by router and head, not by the
        // generic row form the projections use
        S_RSC: begin
            lim = DH;
            ra_k = iw[9:0];
            ra_w = W_ROUT + rt * DM + hh * DH + ia;
        end
        S_RCT: begin
            lim = P;
            ra_pr = ia[9:0];
            ra_v  = hh * DH + ia * DM + o;
        end
        S_TS: begin
            lim = DH;
            ra_q   = iw[9:0];
            ra_rct = rt * DM + hh * DH + ia;
        end
        S_TCT: begin
            lim = NR;
            ra_pr  = ia[9:0];
            ra_rct = hh * DH + ia * DM + o;
        end
        S_RIN: begin
            lim = DM;
            ra_w = W_ROUT + rt * DM + ia;
        end
        S_H1: begin
            lim = DM;
            ra_tok = t * DM + ia;
            ra_q   = t * DM + ia;
        end
        S_LNS, S_LNM, S_LNX, S_LNA: begin
            lim = DM;
            if (ln_bank == B_CTX) ra_ctx = t * DM + ia;
            else                  ra_h2  = t * DM + ia;
        end
        S_GELU, S_FCP: begin
            lim = DFF;
            ra_h2 = t * DFF + ia;
        end
        S_H2A: begin
            lim = DM;
            ra_ctx = t * DM + ia;
            ra_h2  = t * DM + ia;
        end
        S_EMIT: begin
            lim = 16'd1;
            ra_tok = t * HP + o;
        end
        default: ;
        endcase
    end

    wire signed [15:0] opa = (s_bank == B_TOK) ? rd_tok :
                             (s_bank == B_Q)   ? rd_q   :
                             (s_bank == B_K)   ? rd_k   :
                             (s_bank == B_CTX) ? rd_ctx :
                             (s_bank == B_H2)  ? rd_h2  :
                             (s_bank == B_PR)  ? rd_pr  :
                             (s_bank == B_RIN) ? rd_rin :
                             (s_bank == B_XF)  ? rd_xf  : rd_rct;

    wire signed [15:0] opb = (b_kind == 2'd0) ? rd_w :
                             (b_kind == 2'd1) ? rd_v : rd_rct;

    // the layer-norm row and the activation lane both come from whichever bank
    // ln_bank points at, so one read serves both
    wire signed [15:0] ln_rd  = (ln_bank == B_CTX) ? rd_ctx : rd_h2;
    wire signed [15:0] row_now = (i == 0) ? 16'sd0 : ln_rd;

    wire signed [31:0] e_prod = opa * opb;
    wire signed [43:0] e_sum  = (j == 0) ? e_prod : (e_acc + e_prod);

    wire mac_run = (st == S_EMB) || (st == S_Q) || (st == S_K) || (st == S_V)
                || (st == S_RSC) || (st == S_RCT) || (st == S_RPJ)
                || (st == S_TS) || (st == S_TCT) || (st == S_OPJ)
                || (st == S_FF1) || (st == S_FF2) || (st == S_HEAD);

    always @(posedge clk) begin
        if (!rst_n) e_acc <= 0;
        else if (!mac_run || i == 0) e_acc <= 0;
        else e_acc <= e_sum;
    end

    // ---- scratch ----------------------------------------------------------
    reg signed [63:0] acc1, acc2;
    reg signed [31:0] dv_tmp;
    reg signed [63:0] dv_sq;
    reg signed [31:0] ln_mean, ln_inv;
    reg [3:0]  ln_next;
    reg signed [15:0] sm_max;
    reg [31:0] sm_sum, sm_inv;

    wire signed [15:0] gx   = rd_h2;
    wire signed [15:0] gidx = (gx >>> 5) + 16'sd128;
    wire signed [63:0] dev64 = $signed(row_now) - $signed(ln_mean[15:0]);

    // ---- sequencer --------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 0;
            t <= 0; o <= 0; i <= 0; hh <= 0; rt <= 0;
            dv_go <= 0; sq_go <= 0; out_we <= 0; out_addr <= 0; out_data <= 0;
            s_bank <= B_TOK; d_bank <= B_TOK; b_kind <= 0;
            e_nin <= 0; e_nout <= 0; e_ast <= 1; e_soff <= 0;
            e_wb <= 0; e_woff <= 0; e_dast <= DM;
            acc1 <= 0; acc2 <= 0; sm_max <= 0; sm_sum <= 0; sm_inv <= 0;
            ln_mean <= 0; ln_inv <= 0; ln_bank <= B_CTX; ln_next <= 0;
        end else begin
            dv_go <= 1'b0; sq_go <= 1'b0; out_we <= 1'b0;
            if (ld_en) s_xi[ld_addr] <= ld_data;

            case (st)
            S_IDLE: if (start) begin
                st <= S_FOLD; t <= 0; i <= 0; done <= 1'b0;
            end

            // window into phase-major order: one token per phase
            S_FOLD: begin
                s_xf[t * HP + i] <= s_xi[t + i * P];
                if (i == HP-1) begin
                    i <= 0;
                    if (t == P-1) begin
                        st <= S_EMB; t <= 0; o <= 0; i <= 0;
                        s_bank <= B_XF; e_nin <= HP; e_nout <= DM;
                        e_wb <= W_EMB; d_bank <= B_TOK; e_dast <= DM;
                        e_ast <= HP; e_soff <= 0; e_woff <= 0; b_kind <= 0;
                    end else t <= t + 1;
                end else i <= i + 1;
            end

            S_EMB, S_Q, S_K, S_V, S_RPJ, S_OPJ, S_FF1, S_FF2, S_HEAD: begin
                if (i == 0) i <= 1;
                else if (j == e_nin-1) begin
                    case (d_bank)
                    B_TOK: b_tok[t * e_dast + o] <= sat(e_sum);
                    B_Q:   b_q  [t * e_dast + o] <= sat(e_sum);
                    B_K:   b_k  [t * e_dast + o] <= sat(e_sum);
                    B_V:   b_v  [t * e_dast + o] <= sat(e_sum);
                    B_H2:  b_h2 [t * e_dast + o] <= sat(e_sum);
                    B_RCT: s_rct[rt * DM + hh * DH + o] <= sat(e_sum);
                    default: ;
                    endcase
                    i <= 0;
                    if (o == e_nout-1) begin
                        o <= 0;
                        if (t == ((st == S_RPJ) ? 16'd0 : P-1)) begin
                            t <= 0;
                            case (st)
                            S_EMB: begin
                                st <= S_Q; s_bank <= B_TOK; e_nin <= DM; e_nout <= DM;
                                e_wb <= W_Q; d_bank <= B_Q; e_ast <= DM; e_dast <= DM;
                            end
                            S_Q: begin
                                st <= S_K; s_bank <= B_TOK; e_nin <= DM; e_nout <= DM;
                                e_wb <= W_K; d_bank <= B_K; e_ast <= DM; e_dast <= DM;
                            end
                            S_K: begin
                                st <= S_V; s_bank <= B_TOK; e_nin <= DM; e_nout <= DM;
                                e_wb <= W_V; d_bank <= B_V; e_ast <= DM; e_dast <= DM;
                            end
                            S_V: begin
                                st <= S_RSC; hh <= 0; rt <= 0; t <= 0; i <= 0;
                                s_bank <= B_K; e_nin <= DH; e_soff <= 0; e_woff <= 0;
                                e_wb <= W_ROUT; b_kind <= 0; e_ast <= DM;
                            end
                            S_RPJ: begin
                                i <= 0; o <= 0;
                                if (rt != NR-1) begin
                                    rt <= rt + 1; st <= S_RIN;
                                end else begin
                                    rt <= 0; hh <= 0; t <= 0; st <= S_TS;
                                    s_bank <= B_Q; e_nin <= DH; e_soff <= 0; e_ast <= DM;
                                    b_kind <= 2;
                                end
                            end
                            S_OPJ: begin
                                st <= S_H1; t <= 0; i <= 0;
                            end
                            S_FF1: begin st <= S_GELU; t <= 0; i <= 0; end
                            S_FF2: begin
                                st <= S_H2A; t <= 0; i <= 0;
                            end
                            S_HEAD: begin st <= S_EMIT; t <= 0; o <= 0; i <= 0; end
                            default: ;
                            endcase
                        end else t <= t + 1;
                    end else o <= o + 1;
                end else i <= i + 1;
            end

            // router scores for one head and one router
            S_RSC: begin
                if (i == 0) i <= 1;
                else if (j == DH-1) begin
                    s_sc[t] <= sat(e_sum);
                    i <= 0;
                    if (t == P-1) begin
                        t <= 0; st <= S_RSM; sm_max <= -32768; i <= 0;
                    end else t <= t + 1;
                end else i <= i + 1;
            end

            S_RSM: begin
                if (i < P) begin
                    if (s_sc[i] > sm_max) sm_max <= s_sc[i];
                    i <= i + 1;
                end else if (i == P) begin
                    sm_sum <= 0; i <= P + 1;
                end else if (i < 2*P + 1) begin
                    sm_sum <= sm_sum + exp_of(s_sc[i - P - 1] - sm_max);
                    s_pr[i - P - 1] <= exp_of(s_sc[i - P - 1] - sm_max);
                    i <= i + 1;
                end else if (i == 2*P + 1) begin
                    dv_arg <= (1 << 24);
                    dv_div <= (sm_sum == 0) ? 1 : sm_sum;
                    dv_go <= 1'b1; i <= i + 1;
                end else if (i == 2*P + 2) begin
                    if (dv_done) begin sm_inv <= dv_q; i <= i + 1; end
                end else begin
                    s_pr[i - 2*P - 3] <= mulshr32(s_pr[i - 2*P - 3], sm_inv);
                    if (i - (2*P + 3) == P-1) begin
                        // the router context for this head follows immediately:
                        // it only needs the probabilities just written
                        st <= S_RCT; o <= 0; i <= 0;
                        s_bank <= B_PR; e_nin <= P; e_ast <= 0; e_soff <= 0;
                        e_wb <= 0; b_kind <= 1;
                    end else i <= i + 1;
                end
            end

            S_RIN: begin
                if (i == 0) i <= 1;
                else begin
                    s_rin[j] <= rd_w;
                    if (j == DM-1) begin
                        i <= 0; o <= 0; t <= 0;
                        st <= S_RPJ;
                        s_bank <= B_RIN; e_nin <= DM; e_nout <= DM; e_wb <= W_V;
                        d_bank <= B_RCT; b_kind <= 0; e_dast <= DM;
                    end else i <= i + 1;
                end
            end

            // router context for one head
            S_RCT: begin
                if (i == 0) i <= 1;
                else if (j == P-1) begin
                    s_rct[rt * DM + hh * DH + o] <= sat(e_sum);
                    i <= 0;
                    if (o == DH-1) begin
                        o <= 0;
                        if (hh != NH-1) begin
                            hh <= hh + 1; st <= S_RSC; t <= 0;
                            s_bank <= B_K; e_nin <= DH; e_soff <= (hh + 1) * DH;
                            e_ast <= DM;
                        end else if (rt != NR-1) begin
                            hh <= 0; rt <= rt + 1; st <= S_RSC; t <= 0;
                            s_bank <= B_K; e_nin <= DH; e_soff <= 0;
                            e_wb <= W_ROUT + (rt + 1) * DM; e_ast <= DM;
                        end else begin
                            // every router context is in place, but each router
                            // still has to be projected before the token side
                            rt <= 0; hh <= 0; st <= S_RIN; t <= 0; i <= 0; o <= 0;
                        end
                    end else o <= o + 1;
                end else i <= i + 1;
            end

            // token scores over the routers
            S_TS: begin
                if (i == 0) i <= 1;
                else if (j == DH-1) begin
                    s_sc[rt] <= sat(e_sum);
                    i <= 0;
                    if (rt == NR-1) begin
                        rt <= 0; st <= S_TSM; sm_max <= -32768; i <= 0;
                    end else rt <= rt + 1;
                end else i <= i + 1;
            end

            S_TSM: begin
                if (i < NR) begin
                    if (s_sc[i] > sm_max) sm_max <= s_sc[i];
                    i <= i + 1;
                end else if (i == NR) begin
                    sm_sum <= 0; i <= NR + 1;
                end else if (i < 2*NR + 1) begin
                    sm_sum <= sm_sum + exp_of(s_sc[i - NR - 1] - sm_max);
                    s_pr[i - NR - 1] <= exp_of(s_sc[i - NR - 1] - sm_max);
                    i <= i + 1;
                end else if (i == 2*NR + 1) begin
                    dv_arg <= (1 << 24);
                    dv_div <= (sm_sum == 0) ? 1 : sm_sum;
                    dv_go <= 1'b1; i <= i + 1;
                end else if (i == 2*NR + 2) begin
                    if (dv_done) begin sm_inv <= dv_q; i <= i + 1; end
                end else begin
                    s_pr[i - 2*NR - 3] <= mulshr32(s_pr[i - 2*NR - 3], sm_inv);
                    if (i - (2*NR + 3) == NR-1) begin
                        st <= S_TCT; o <= 0; i <= 0;
                        s_bank <= B_PR; e_nin <= NR; e_ast <= 0; e_soff <= 0;
                        e_wb <= 0; b_kind <= 2;
                    end else i <= i + 1;
                end
            end

            // token context for one head
            S_TCT: begin
                if (i == 0) i <= 1;
                else if (j == NR-1) begin
                    b_ctx[t * DM + hh * DH + o] <= sat(e_sum);
                    i <= 0;
                    if (o == DH-1) begin
                        o <= 0;
                        if (t == P-1) begin
                            if (hh == NH-1) begin
                                hh <= 0; t <= 0; st <= S_OPJ; o <= 0; i <= 0;
                                // the projection cannot write back into its own
                                // source bank: output o would clobber input o
                                s_bank <= B_CTX; e_nin <= DM; e_nout <= DM;
                                e_wb <= W_O; d_bank <= B_Q; e_ast <= DM;
                                e_dast <= DM; b_kind <= 0;
                            end else begin
                                // a new head needs its own scores, so the token
                                // side starts again rather than reusing these
                                hh <= hh + 1; t <= 0; rt <= 0; i <= 0; st <= S_TS;
                                s_bank <= B_Q; e_nin <= DH; e_soff <= (hh + 1) * DH;
                                e_ast <= DM;
                            end
                        end else begin
                            // the next token needs its own scores first, with the
                            // engine pointed back at the queries
                            t <= t + 1; rt <= 0; i <= 0; st <= S_TS;
                            s_bank <= B_Q; e_nin <= DH; e_soff <= hh * DH;
                            e_ast <= DM;
                        end
                    end else o <= o + 1;
                end else i <= i + 1;
            end

            // h1 = token + attention output, into the context bank
            S_H1: begin
                if (i == 0) i <= 1;
                else begin
                    b_ctx[t * DM + j] <= sat(($signed(rd_tok) + $signed(rd_q)) <<< WQ);
                    if (j == DM-1) begin
                        i <= 0;
                        if (t == P-1) begin
                            t <= 0; st <= S_LNS; i <= 0; acc1 <= 0; acc2 <= 0;
                            ln_bank <= B_CTX; ln_next <= 0;
                        end else t <= t + 1;
                    end else i <= i + 1;
                end
            end

            // layer norm: sum, mean, variance, scale, then apply
            S_LNS: begin
                if (i == 0) i <= 1;
                else begin
                    acc1 <= acc1 + row_now;
                    if (j == DM-1) begin
                        i <= 0; st <= S_LNM;
                        dv_neg <= (acc1 + row_now) < 0;
                        dv_arg <= (acc1 + row_now) < 0 ? $unsigned(-(acc1 + row_now))
                                                       : $unsigned(acc1 + row_now);
                        dv_div <= DM; dv_go <= 1'b1;
                    end else i <= i + 1;
                end
            end

            S_LNM: if (dv_done) begin
                ln_mean <= dv_neg ? -$signed(dv_q) : $signed(dv_q);
                acc2 <= 0; i <= 0; st <= S_LNX;
            end

            // variance around the mean, as the reference computes it
            S_LNX: begin
                if (i == 0) i <= 1;
                else begin
                    acc2 <= acc2 + (dev64 * dev64);
                    if (j == DM-1) begin
                        i <= 0; st <= S_LNQ;
                        dv_arg <= acc2[39:0] + (dev64 * dev64);
                        dv_div <= DM; dv_go <= 1'b1;
                    end else i <= i + 1;
                end
            end

            S_LNQ: if (dv_done) begin
                sq_arg <= ($signed(dv_q) > 0) ? dv_q : 40'd0;
                sq_go <= 1'b1; st <= S_LNI;
            end

            S_LNI: if (sq_done) begin
                dv_arg <= 32'h0100_0000;
                dv_div <= (sq_res > FLR) ? sq_res : FLR;
                dv_go <= 1'b1; st <= S_LNV;
            end

            S_LNA: begin
                if (i == 0) i <= 1;
                else begin
                    if (ln_bank == B_CTX)
                        b_ctx[t * DM + j] <= mulshr32(rd_ctx - ln_mean[15:0], ln_inv);
                    else
                        b_h2[t * DM + j] <= mulshr32(rd_h2 - ln_mean[15:0], ln_inv);
                    if (j == DM-1) begin
                        i <= 0;
                        if (t == P-1) begin
                            t <= 0;
                            if (ln_next == 0) begin
                                st <= S_FF1; o <= 0;
                                s_bank <= B_CTX; e_nin <= DM; e_nout <= DFF;
                                e_wb <= W_F1; d_bank <= B_H2; e_ast <= DM;
                                e_dast <= DM; b_kind <= 0;
                            end else begin
                                st <= S_HEAD; o <= 0;
                                s_bank <= B_H2; e_nin <= DM; e_nout <= HP;
                                e_wb <= W_HEAD; d_bank <= B_TOK; e_ast <= DM;
                                e_dast <= HP; b_kind <= 0;
                            end
                        end else begin
                            // each token has its own statistics
                            t <= t + 1; i <= 0; st <= S_LNS;
                            acc1 <= 0; acc2 <= 0;
                        end
                    end else i <= i + 1;
                end
            end

            S_LNV: if (dv_done) begin
                ln_inv <= dv_q; i <= 0; st <= S_LNA;
            end

            S_GELU: begin
                if (i == 0) i <= 1;
                else begin
                    b_h2[t * DFF + j] <=
                        gtab[(gidx < 0) ? 0 : ((gidx > 15'sd255) ? 255 : gidx[7:0])];
                    if (j == DFF-1) begin
                        i <= 0;
                        if (t == P-1) begin
                            t <= 0; st <= S_FCP; i <= 0;
                        end else t <= t + 1;
                    end else i <= i + 1;
                end
            end

            // move the activated row aside so the projection can write in place
            S_FCP: begin
                if (i == 0) i <= 1;
                else begin
                    b_q[t * DFF + j] <= rd_h2;
                    if (j == DFF-1) begin
                        i <= 0;
                        if (t == P-1) begin
                            t <= 0; o <= 0;
                            st <= S_FF2;
                            s_bank <= B_Q; e_nin <= DFF; e_nout <= DM;
                            e_wb <= W_F2; d_bank <= B_H2; e_ast <= DFF;
                            e_dast <= DM; b_kind <= 0;
                        end else t <= t + 1;
                    end else i <= i + 1;
                end
            end

            // h2 = h1 + projection, then normalise again
            S_H2A: begin
                if (i == 0) i <= 1;
                else begin
                    b_h2[t * DM + j] <= sat(($signed(rd_ctx) + $signed(rd_h2)) <<< WQ);
                    if (j == DM-1) begin
                        i <= 0;
                        if (t == P-1) begin
                            t <= 0; st <= S_LNS; i <= 0; acc1 <= 0; acc2 <= 0;
                            ln_bank <= B_H2; ln_next <= 1;
                        end else t <= t + 1;
                    end else i <= i + 1;
                end
            end

            S_EMIT: begin
                if (i == 0) i <= 1;
                else begin
                    out_addr <= o * P + t;
                    out_data <= rd_tok;
                    out_we   <= 1'b1;
                    i <= 0;
                    if (o == HP-1) begin
                        o <= 0;
                        if (t == P-1) begin st <= S_DONE; done <= 1'b1; end
                        else t <= t + 1;
                    end else o <= o + 1;
                end
            end

            S_DONE: ;
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
