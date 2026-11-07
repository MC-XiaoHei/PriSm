// Drives phaseformer_top through one forecast and compares against fits_ref.py.
`timescale 1ns / 1ps

module tb_pf;
    localparam integer L = 96, H = 96;

    reg clk = 0, rst_n = 0, start = 0, ld_en = 0;
    reg [7:0]  ld_addr = 0;
    reg signed [15:0] ld_data = 0;
    wire done;
    wire signed [15:0] out_data;
    wire [7:0]  out_addr;
    wire out_we;

    reg [15:0] xq  [0:L-1];
    reg [15:0] yref[0:H-1];
    reg signed [15:0] got [0:H-1];
    integer i, bad;

    initial begin
        $readmemh("pf_x_in.hex", xq);
        $readmemh("pf_y_ref.hex", yref);
    end

        initial begin
        `ifdef TRACE
        $dumpfile("pf.vcd");
        $dumpvars(0, tb_pf);
        `endif
    end

    phaseformer_top #(.L(L)) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .ld_en(ld_en), .ld_addr(ld_addr), .ld_data(ld_data),
        .done(done),
        .out_data(out_data), .out_addr(out_addr), .out_we(out_we));

    always #5 clk = ~clk;

    always @(posedge clk) if (out_we) got[out_addr] = out_data;

    // The design's own counter is not a port any more, so the bench measures
    // the active cycles itself: from the edge that samples start up to the edge
    // that samples done.
    integer cyc = 0;
    reg run = 0;
    always @(posedge clk) begin
        if (start) begin run <= 1; cyc <= 0; end
        else if (run) begin
            if (done) run <= 0;
            else cyc <= cyc + 1;
        end
    end

    integer dq = 0;
    integer visits [0:31];
    integer vi;
`ifndef GATE_SIM
    initial for (vi = 0; vi < 32; vi = vi + 1) visits[vi] = 0;
    always @(posedge clk) visits[dut.st] = visits[dut.st] + 1;
    integer sc0 = 0;
    always @(posedge clk) if (dut.st == 5'd7 && sc0 < 1) begin
        sc0 = 1;
        $writememh("pf_dump_scores_tok.hex", dut.b_tok);
        $writememh("pf_dump_scores_q.hex", dut.b_q);
        $writememh("pf_dump_scores_k.hex", dut.b_k);
        $writememh("pf_dump_scores_v.hex", dut.b_v);
        $writememh("pf_dump_scores_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_scores_h2.hex", dut.b_h2);
        $writememh("pf_dump_scores_sc.hex", dut.s_sc);
        $writememh("pf_dump_scores_pr.hex", dut.s_pr);
        $writememh("pf_dump_scores_rct.hex", dut.s_rct);
    end
`endif
    integer sc1 = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd8 && sc1 < 1) begin
        sc1 = 1;
        $writememh("pf_dump_probs_tok.hex", dut.b_tok);
        $writememh("pf_dump_probs_q.hex", dut.b_q);
        $writememh("pf_dump_probs_k.hex", dut.b_k);
        $writememh("pf_dump_probs_v.hex", dut.b_v);
        $writememh("pf_dump_probs_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_probs_h2.hex", dut.b_h2);
        $writememh("pf_dump_probs_sc.hex", dut.s_sc);
        $writememh("pf_dump_probs_pr.hex", dut.s_pr);
        $writememh("pf_dump_probs_rct.hex", dut.s_rct);
    end
`endif
    integer sc7 = 0;
    integer dg = 0;
    integer dg2 = 0;
    integer dg3 = 0;
    reg [4:0] last_st = 31;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st != last_st && sc7 < 40) begin
        last_st = dut.st; sc7 = sc7 + 1;
        $display("TR %0d t=%0d i=%0d o=%0d nin=%0d nout=%0d ast=%0d bank=%0d bkind=%0d",
                 dut.st, dut.t, dut.i, dut.o, dut.e_nin, dut.e_nout, dut.e_ast, dut.s_bank, dut.b_kind);
    end
`endif
    integer sc6 = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd27 && dut.i == 31 && sc6 < 6) begin
        sc6 = sc6 + 1;
        $display("LNA i=%0d t=%0d bank=%0d inv=%0d mean=%0d", dut.i, dut.t, dut.ln_bank, $signed(dut.ln_inv), $signed(dut.ln_mean));
    end
`endif
    integer scv = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd25 && dut.dv_done && scv < 6) begin
        scv = scv + 1;
        $display("LNQV t=%0d q=%0d arg=%0d div=%0d", dut.t, $signed(dut.dv_q), $signed(dut.dv_arg), dut.dv_div);
    end
`endif
    integer sc5 = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd22 && sc5 == 0 && dut.ln_bank == 4) begin
        sc5 = 1;
        $writememh("pf_dump_h1pre_tok.hex", dut.b_tok);
        $writememh("pf_dump_h1pre_q.hex", dut.b_q);
        $writememh("pf_dump_h1pre_k.hex", dut.b_k);
        $writememh("pf_dump_h1pre_v.hex", dut.b_v);
        $writememh("pf_dump_h1pre_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_h1pre_h2.hex", dut.b_h2);
        $writememh("pf_dump_h1pre_sc.hex", dut.s_sc);
        $writememh("pf_dump_h1pre_pr.hex", dut.s_pr);
        $writememh("pf_dump_h1pre_rct.hex", dut.s_rct);
    end
`endif
    integer sc4 = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd11 && dut.t == 1 && sc4 < 10) begin
        sc4 = sc4 + 1;
        $display("EN i=%0d rt=%0d ast=%0d soff=%0d src=%0d wb=%0d a=%0d b=%0d",
            dut.i, dut.rt, dut.e_ast, dut.e_soff, dut.s_bank, dut.e_wb, $signed(dut.opa), $signed(dut.opb));
    end
`endif
    integer sc3 = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd12 && dut.t == 1 && sc3 < 1) begin
        sc3 = 1;
        $writememh("pf_dump_ts_tok.hex", dut.b_tok);
        $writememh("pf_dump_ts_q.hex", dut.b_q);
        $writememh("pf_dump_ts_k.hex", dut.b_k);
        $writememh("pf_dump_ts_v.hex", dut.b_v);
        $writememh("pf_dump_ts_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_ts_h2.hex", dut.b_h2);
        $writememh("pf_dump_ts_sc.hex", dut.s_sc);
        $writememh("pf_dump_ts_pr.hex", dut.s_pr);
        $writememh("pf_dump_ts_rct.hex", dut.s_rct);
    end
`endif
    integer sc2 = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd11 && sc2 < 1) begin
        sc2 = 1;
        $writememh("pf_dump_rct_tok.hex", dut.b_tok);
        $writememh("pf_dump_rct_q.hex", dut.b_q);
        $writememh("pf_dump_rct_k.hex", dut.b_k);
        $writememh("pf_dump_rct_v.hex", dut.b_v);
        $writememh("pf_dump_rct_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_rct_h2.hex", dut.b_h2);
        $writememh("pf_dump_rct_sc.hex", dut.s_sc);
        $writememh("pf_dump_rct_pr.hex", dut.s_pr);
        $writememh("pf_dump_rct_rct.hex", dut.s_rct);
    end
`endif
    integer tc = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd13 && tc < 10) begin
        tc = tc + 1;
        $display("TCT i=%0d j=%0d o=%0d hh=%0d t=%0d a=%0d b=%0d sum=%0d",
            dut.i, dut.j, dut.o, dut.hh, dut.t, $signed(dut.opa), $signed(dut.opb), $signed(dut.e_sum));
    end
`endif
    integer gp = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd17 && gp < 8) begin
        gp = gp + 1;
        $display("GELU i=%0d j=%0d ra=%0d b0=%0d b1=%0d rdh2=%0d x=%0d",
            dut.i, dut.j, dut.ra_h2, $signed(dut.b_h2[0]), $signed(dut.b_h2[1]),
            $signed(dut.rd_h2), $signed(dut.gx));
    end
`endif
    integer rp = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd10 && rp < 8) begin
        rp = rp + 1;
        $display("NEW RPJ i=%0d j=%0d o=%0d soff=%0d a=%0d b=%0d sum=%0d",
            dut.i, dut.j, dut.o, dut.e_soff, $signed(dut.opa), $signed(dut.opb), $signed(dut.e_sum));
    end
`endif
    integer rs = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd7 && rs == 0) begin
        rs = 1;
        $writememh("new_sc.hex", dut.s_sc);
        $writememh("new_pr.hex", dut.s_pr);
    end
`endif
    integer rq = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd8 && rq == 0) begin
        rq = 1;
        $writememh("new_rct8.hex", dut.s_rct);
    end
`endif
    integer rx = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd6 && rx == 0) begin
        rx = 1;
        $writememh("pf_dump_rsc_tok.hex", dut.b_tok);
        $writememh("pf_dump_rsc_q.hex", dut.b_q);
        $writememh("pf_dump_rsc_k.hex", dut.b_k);
        $writememh("pf_dump_rsc_v.hex", dut.b_v);
        $writememh("pf_dump_rsc_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_rsc_h2.hex", dut.b_h2);
        $writememh("pf_dump_rsc_sc.hex", dut.s_sc);
        $writememh("pf_dump_rsc_pr.hex", dut.s_pr);
        $writememh("pf_dump_rsc_rct.hex", dut.s_rct);
    end
`endif
    integer vp = 0;
    always @(posedge clk) if (done && vp < 1) begin
        vp = 1;
        for (vi = 0; vi < 32; vi = vi + 1)
            if (visits[vi] > 0) $display("VISIT %0d %0d", vi, visits[vi]);
    end
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd3 && dq < 1) begin
        dq = 1;
        $writememh("pf_dump_tok_tok.hex", dut.b_tok);
        $writememh("pf_dump_tok_q.hex", dut.b_q);
        $writememh("pf_dump_tok_k.hex", dut.b_k);
        $writememh("pf_dump_tok_v.hex", dut.b_v);
        $writememh("pf_dump_tok_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_tok_h2.hex", dut.b_h2);
        $writememh("pf_dump_tok_sc.hex", dut.s_sc);
        $writememh("pf_dump_tok_pr.hex", dut.s_pr);
        $writememh("pf_dump_tok_rct.hex", dut.s_rct);
    end
`endif
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd4 && dq == 1) begin
        dq = 2;
        $writememh("pf_dump_q_tok.hex", dut.b_tok);
        $writememh("pf_dump_q_q.hex", dut.b_q);
        $writememh("pf_dump_q_k.hex", dut.b_k);
        $writememh("pf_dump_q_v.hex", dut.b_v);
        $writememh("pf_dump_q_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_q_h2.hex", dut.b_h2);
        $writememh("pf_dump_q_sc.hex", dut.s_sc);
        $writememh("pf_dump_q_pr.hex", dut.s_pr);
        $writememh("pf_dump_q_rct.hex", dut.s_rct);
    end
`endif
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd14 && dq == 2) begin
        dq = 3;
        $writememh("pf_dump_ctx_tok.hex", dut.b_tok);
        $writememh("pf_dump_ctx_q.hex", dut.b_q);
        $writememh("pf_dump_ctx_k.hex", dut.b_k);
        $writememh("pf_dump_ctx_v.hex", dut.b_v);
        $writememh("pf_dump_ctx_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_ctx_h2.hex", dut.b_h2);
        $writememh("pf_dump_ctx_sc.hex", dut.s_sc);
        $writememh("pf_dump_ctx_pr.hex", dut.s_pr);
        $writememh("pf_dump_ctx_rct.hex", dut.s_rct);
    end
`endif
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd16 && dq == 3) begin
        dq = 4;
        $writememh("pf_dump_h1_tok.hex", dut.b_tok);
        $writememh("pf_dump_h1_q.hex", dut.b_q);
        $writememh("pf_dump_h1_k.hex", dut.b_k);
        $writememh("pf_dump_h1_v.hex", dut.b_v);
        $writememh("pf_dump_h1_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_h1_h2.hex", dut.b_h2);
        $writememh("pf_dump_h1_sc.hex", dut.s_sc);
        $writememh("pf_dump_h1_pr.hex", dut.s_pr);
        $writememh("pf_dump_h1_rct.hex", dut.s_rct);
    end
`endif
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd17 && dg == 0) begin
        dg = 1; $writememh("pf_dump_f1_tok.hex", dut.b_tok);
        $writememh("pf_dump_f1_q.hex", dut.b_q);
        $writememh("pf_dump_f1_k.hex", dut.b_k);
        $writememh("pf_dump_f1_v.hex", dut.b_v);
        $writememh("pf_dump_f1_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_f1_h2.hex", dut.b_h2);
        $writememh("pf_dump_f1_sc.hex", dut.s_sc);
        $writememh("pf_dump_f1_pr.hex", dut.s_pr);
        $writememh("pf_dump_f1_rct.hex", dut.s_rct);
    end
`endif
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd18 && dg == 1 && dg2 == 0) begin
        dg2 = 1; $writememh("pf_dump_f1a_tok.hex", dut.b_tok);
        $writememh("pf_dump_f1a_q.hex", dut.b_q);
        $writememh("pf_dump_f1a_k.hex", dut.b_k);
        $writememh("pf_dump_f1a_v.hex", dut.b_v);
        $writememh("pf_dump_f1a_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_f1a_h2.hex", dut.b_h2);
        $writememh("pf_dump_f1a_sc.hex", dut.s_sc);
        $writememh("pf_dump_f1a_pr.hex", dut.s_pr);
        $writememh("pf_dump_f1a_rct.hex", dut.s_rct);
    end
`endif
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd20 && dg2 == 1 && dg3 == 0) begin
        dg3 = 1; $writememh("pf_dump_h2pre_tok.hex", dut.b_tok);
        $writememh("pf_dump_h2pre_q.hex", dut.b_q);
        $writememh("pf_dump_h2pre_k.hex", dut.b_k);
        $writememh("pf_dump_h2pre_v.hex", dut.b_v);
        $writememh("pf_dump_h2pre_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_h2pre_h2.hex", dut.b_h2);
        $writememh("pf_dump_h2pre_sc.hex", dut.s_sc);
        $writememh("pf_dump_h2pre_pr.hex", dut.s_pr);
        $writememh("pf_dump_h2pre_rct.hex", dut.s_rct);
    end
`endif
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd21 && dq == 4) begin
        dq = 5;
        $writememh("pf_dump_h2_tok.hex", dut.b_tok);
        $writememh("pf_dump_h2_q.hex", dut.b_q);
        $writememh("pf_dump_h2_k.hex", dut.b_k);
        $writememh("pf_dump_h2_v.hex", dut.b_v);
        $writememh("pf_dump_h2_ctx.hex", dut.b_ctx);
        $writememh("pf_dump_h2_h2.hex", dut.b_h2);
        $writememh("pf_dump_h2_sc.hex", dut.s_sc);
        $writememh("pf_dump_h2_pr.hex", dut.s_pr);
        $writememh("pf_dump_h2_rct.hex", dut.s_rct);
    end
`endif

`ifndef GATE_SIM
    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        for (i = 0; i < L; i = i + 1) begin
            ld_en <= 1; ld_addr <= i[7:0]; ld_data <= xq[i];
            @(posedge clk);
        end
        ld_en <= 0;
        @(posedge clk);
        start <= 1; @(posedge clk); start <= 0;

        for (i = 0; i < 4000000 && !done; i = i + 1) @(posedge clk);
        if (!done) begin
            $display("TIMEOUT st=%0d i=%0d t=%0d hh=%0d rt=%0d o=%0d cyc=%0d",
                     dut.st, dut.i, dut.t, dut.hh, dut.rt, dut.o, cyc);
            for (i = 0; i < 32; i = i + 1)
                if (visits[i] > 0) $display("VISIT %0d %0d", i, visits[i]);
            $finish;
        end
        repeat (4) @(posedge clk);

        bad = 0;
        for (i = 0; i < H; i = i + 1)
            if (got[i] !== $signed(yref[i])) begin
                bad = bad + 1;
                if (bad < 6)
                    $display("MISMATCH i=%0d got=%0d want=%0d", i, got[i], $signed(yref[i]));
            end
        $display("cycles=%0d mismatches=%0d of %0d", cyc, bad, H);
        $finish;
    end
`endif
`ifdef TRACE_STOP
    always @(posedge clk) if ($time >= `TRACE_STOP) $finish;
`endif

endmodule
