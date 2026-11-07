// Compare timemixer_top against the integer reference.
`timescale 1ns / 1ps

module tb_tm;
    localparam integer L    = 96;
    localparam integer PRED = 96;
    localparam integer WROM = 75825;

    reg clk = 0, rst_n = 0, start = 0, ld_en = 0;
    reg [1:0]  ld_sel = 0;
    reg [16:0] ld_addr = 0;
    reg [15:0] ld_data = 0;
    reg [3:0]  ld_sub = 0;
    wire done;
    wire signed [15:0] rd_data;

    always #5 clk = ~clk;

    initial begin
        `ifdef TRACE
        $dumpfile("tm.vcd");
        $dumpvars(0, tb_tm);
        `endif
    end

    timemixer_top dut (
        .clk(clk), .rst_n(rst_n), .start(start), .ld_en(ld_en), .ld_sel(ld_sel),
        .ld_addr(ld_addr), .ld_data(ld_data), .ld_sub(ld_sub), .done(done),
        .rd_addr(7'd0), .rd_data(rd_data)
    );

    reg [15:0]  wimg [0:WROM-1];
    reg [255:0] gimg [0:255];
    reg [15:0]  xin  [0:L-1];
    reg signed [15:0] want [0:PRED-1];
    reg signed [15:0] got  [0:PRED-1];

    integer i, g, mism, guard;

    // The design reports done, not a cycle count, so the bench measures the
    // active cycles itself: from the edge that samples start to the edge that
    // samples done.
    integer cyc = 0;
    reg run = 0;
    always @(posedge clk) begin
        if (start) begin run <= 1; cyc <= 0; end
        else if (run) begin
            if (done) run <= 0;
            else cyc <= cyc + 1;
        end
    end

`ifndef GATE_SIM
    initial begin
        $readmemh("tm_w.hex", wimg);
        $readmemh("tm_g.hex", gimg);
        $readmemh("tm_x.hex", xin);
        $readmemh("tm_y_ref.hex", want);
        @(negedge clk);
        rst_n = 1;

        for (i = 0; i < WROM; i = i + 1) begin
            @(negedge clk);
            ld_en = 1; ld_sel = 2'd0; ld_addr = i[16:0]; ld_data = wimg[i];
        end
        for (i = 0; i < 256; i = i + 1) begin
            for (g = 0; g < 16; g = g + 1) begin
                @(negedge clk);
                ld_en = 1; ld_sel = 2'd1; ld_addr = i[16:0];
                ld_sub = g[3:0]; ld_data = gimg[i][g*16 +: 16];
            end
        end
        for (i = 0; i < L; i = i + 1) begin
            @(negedge clk);
            ld_en = 1; ld_sel = 2'd2; ld_addr = i[16:0]; ld_data = xin[i];
        end
        @(negedge clk);
        ld_en = 0;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        guard = 0;
        while (!done && guard < 8000000) begin
            @(negedge clk);
            guard = guard + 1;
        end
        @(negedge clk);
        @(negedge clk);
        if (!done) begin
            $display("TIMEOUT after %0d cycles, state %0d", guard, dut.st);
            $finish;
        end

        for (i = 0; i < PRED; i = i + 1) got[i] = dut.outv[i];

        mism = 0;
        for (i = 0; i < PRED; i = i + 1) begin
            if (got[i] !== want[i]) begin
                if (mism < 8) $display("MISMATCH i=%0d got=%0d want=%0d", i, got[i], want[i]);
                mism = mism + 1;
            end
        end
        $display("cycles=%0d mismatches=%0d of %0d", cyc, mism, PRED);
        $finish;
    end
`endif
`ifdef TRACE_STOP
    always @(posedge clk) if ($time >= `TRACE_STOP) $finish;
`endif

endmodule
