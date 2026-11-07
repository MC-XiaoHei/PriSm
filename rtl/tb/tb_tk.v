// Compare timekan_top against the integer reference.
`timescale 1ns / 1ps

module tb_tk;
    localparam integer L = 96;
    localparam integer WMEM = 18880;

    reg clk = 0, rst_n = 0, start = 0, ld_en = 0;
    reg [1:0] ld_sel = 0;
    reg [15:0] ld_addr = 0;
    reg signed [15:0] ld_data = 0;
    wire done;
    wire [15:0] rd_addr;
    wire signed [15:0] rd_data;

    always #5 clk = ~clk;

    initial begin
        `ifdef TRACE
        $dumpfile("tk.vcd");
        $dumpvars(0, tb_tk);
        `endif
    end

    timekan_top dut (
        .clk(clk), .rst_n(rst_n), .start(start), .ld_en(ld_en), .ld_sel(ld_sel),
        .ld_addr(ld_addr), .ld_data(ld_data), .done(done),
        .rd_addr(rd_addr), .rd_data(rd_data)
    );

    reg signed [15:0] wimg [0:WMEM-1];
    reg signed [15:0] xin  [0:L-1];
    reg signed [15:0] want [0:L-1];
    reg signed [15:0] got  [0:L-1];

    integer i, mism, guard;
    reg signed [15:0] tmpmem [0:4095];

`ifndef GATE_SIM
    task copyx;
        input integer base;
        input integer n;
        integer j;
        begin
            for (j = 0; j < n; j = j + 1) tmpmem[j] = dut.xmem[base + j];
        end
    endtask

    task copybank;
        input integer base;
        input integer n;
        integer j;
        begin
            for (j = 0; j < n; j = j + 1) tmpmem[j] = dut.bmem[base + j];
        end
    endtask
`endif


`ifndef GATE_SIM
    always @(posedge clk) if (dut.out_we) got[dut.out_addr] = dut.outv;

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
`endif

    integer d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0, d8 = 0, d9 = 0;
`ifndef GATE_SIM
    always @(posedge clk) begin
        if (dut.st == 5'd15 && d1 == 0) begin d1 = 1; copybank(3360, 1536); $writememh("rtl_e1_raw.hex", tmpmem); end
        if (dut.st == 5'd16 && d2 == 0) begin d2 = 1; copybank(11040, 3072); $writememh("rtl_up_raw.hex", tmpmem); end
        if (dut.st == 5'd18 && d3 == 0) begin d3 = 1; copybank(4896, 3072); $writememh("rtl_hi_raw.hex", tmpmem); end
        if (dut.st == 5'd17 && d4 == 0) begin d4 = 1; copybank(7968, 1536); $writememh("rtl_lowmix.hex", tmpmem); end
        if (dut.st == 5'd18 && d5 == 0) begin d5 = 1; copybank(3360, 1536); $writememh("rtl_lowmix2.hex", tmpmem); end
        if (dut.st == 5'd19 && d6 == 0) begin d6 = 1; copybank(7968, 3072); $writememh("rtl_hikan.hex", tmpmem); end
        if (dut.st == 5'd20 && d7 == 0) begin d7 = 1; copybank(4896, 3072); $writememh("rtl_highmix.hex", tmpmem); end
        if (dut.st == 5'd22 && d8 == 0) begin d8 = 1; copybank(7968, 3072); $writememh("rtl_mixed.hex", tmpmem); end
        if (dut.st == 5'd23 && d9 == 0) begin d9 = 1; copybank(7968, 3072); $writememh("rtl_dense.hex", tmpmem); end
    end
`endif

    integer pz = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd1 && pz < 3) begin
        pz = pz + 1;
        $display("BIAS pb0=%0d pb1=%0d pb95=%0d proj=%0d", $signed(dut.pbias[0]), $signed(dut.pbias[1]), $signed(dut.pbias[95]), $signed(dut.projbias));
    end
`endif
    integer pq = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd18 && dut.t == 0 && dut.i == 0 && pq < 8) begin
        pq = pq + 1;
        $display("BAS bp=%0d tbl=%0d qa=%0d qb=%0d", dut.bp, dut.tbl_addr, $signed(dut.q_a), $signed(dut.q_b));
    end
`endif
    integer pe = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd13 && dut.t == 0 && dut.o == 0 && pe < 5) begin
        pe = pe + 1;
        $display("E1 i=%0d idx=%0d n1=%0d wb=%0d x=%0d", dut.i,
                 (dut.t + dut.i + 48 - 2) % 48,
                 $signed(dut.xmem[240 + ((dut.t + dut.i + 48 - 2) % 48)]),
                 $signed(dut.q_b), $signed(dut.xmem[192 + dut.t]));
    end
`endif
    integer pd = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd24 && pd < 4) begin
        pd = pd + 1;
        $display("DEN t=%0d out_word=%0d std=%0d mean=%0d outv=%0d", dut.t, $signed(dut.out_word), $signed(dut.lvl0_std), $signed(dut.lvl0_mean), $signed(dut.outv));
    end
`endif
    integer db = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st == 5'd27 && dut.t == 0 && db == 0) begin
        db = 1;
        for (i = 0; i < 32; i = i + 1) tmpmem[i] = dut.bx1[i];
        $writememh("rtl_bx1.hex", tmpmem);
    end
`endif
    integer cw = 0, cw2 = 0, vb = 0, vm = 0, vc = 0;
`ifndef GATE_SIM
    always @(posedge clk) begin
        if (dut.st == 5'd18) vb = vb + 1;
        if (dut.st == 5'd27) vm = vm + 1;
        if (dut.st == 5'd19) vc = vc + 1;
    end
`endif
`ifndef GATE_SIM
    always @(posedge clk) begin
        if (dut.st == 5'd17 && dut.i == 3) cw = cw + 1;
        if (dut.st == 5'd17) cw2 = cw2 + 1;
    end
`endif

    reg [4:0] last_st = 31;
    integer tr = 0;
`ifndef GATE_SIM
    always @(posedge clk) if (dut.st != last_st && tr < 60) begin
        last_st = dut.st; tr = tr + 1;
        $display("TR cyc=%0d %0d -> %0d t=%0d o=%0d i=%0d ph=%0d", cyc, last_st, dut.st, dut.t, dut.o, dut.i, dut.ph);
    end
`endif

`ifndef GATE_SIM
    initial begin
        $readmemh("tk_w.hex", wimg);
        $readmemh("tk_x.hex", xin);
        $readmemh("tk_y_ref.hex", want);
        @(negedge clk);
        rst_n = 1;

        for (i = 0; i < WMEM; i = i + 1) begin
            @(negedge clk);
            ld_en = 1; ld_sel = 2'd0; ld_addr = i[15:0]; ld_data = wimg[i];
        end
        for (i = 0; i < L; i = i + 1) begin
            @(negedge clk);
            ld_en = 1; ld_sel = 2'd2; ld_addr = i[15:0]; ld_data = xin[i];
        end
        @(negedge clk);
        ld_en = 0;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        guard = 0;
        while (!done && guard < 4000000) begin
            @(negedge clk);
            guard = guard + 1;
        end
        @(negedge clk);
        @(negedge clk);
        if (!done) begin
            $display("TIMEOUT after %0d cycles, state %0d", guard, dut.st);
            $writememh("tk_final_bmem.hex", dut.bmem);
        $finish;
        end


        copyx(96, 96);        $writememh("rtl_n0.hex", tmpmem);
        copyx(240, 48);       $writememh("rtl_n1.hex", tmpmem);
        copyx(192, 48);       $writememh("rtl_l1.hex", tmpmem);
        copybank(288, 3072);     $writememh("rtl_e0.hex", tmpmem);
        copybank(3360, 1536);    $writememh("rtl_e1.hex", tmpmem);
        copybank(11040, 1536);   $writememh("rtl_lowband.hex", tmpmem);
        copybank(12576, 3072);   $writememh("rtl_highband.hex", tmpmem);
        copybank(15648, 3072);   $writememh("rtl_mix.hex", tmpmem);
        $writememh("tk_final_bmem.hex", dut.bmem);

        mism = 0;
        for (i = 0; i < L; i = i + 1) begin
            if (got[i] !== want[i]) begin
                if (mism < 6) $display("MISMATCH i=%0d got=%0d want=%0d", i, got[i], want[i]);
                mism = mism + 1;
            end
        end
        $display("cycles=%0d mismatches=%0d of %0d  conlv_cycles=%0d writes=%0d basis=%0d mixh=%0d convh=%0d", cyc, mism, L, cw2, cw, vb, vm, vc);
        $finish;
    end
`endif
`ifdef TRACE_STOP
    always @(posedge clk) if ($time >= `TRACE_STOP) $finish;
`endif

endmodule
