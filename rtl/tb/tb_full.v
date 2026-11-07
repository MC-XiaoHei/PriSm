// Drives prism_top through one forecast and compares the 96 outputs against
// full_ref.py's answer. Self-contained: the clock and the load sequence live
// here, so it builds with --binary --timing.
`timescale 1ns / 1ps

module tb_full;
    localparam integer L = 96, P = 24, HP = 4;

    reg clk = 0, rst_n = 0, start = 0, ld_en = 0;
    reg [7:0] ld_addr = 0;
    reg signed [15:0] ld_data = 0;
    wire done;
    wire signed [15:0] out_data;
    wire [7:0] out_addr;
    wire out_we;

    reg [15:0] xq  [0:L-1];
    reg [15:0] yref[0:HP*P-1];
    reg signed [15:0] got [0:HP*P-1];
    integer i, bad;

    initial begin
        `ifdef TRACE
        $dumpfile("prism.vcd");
        $dumpvars(0, tb_full);
        `endif
        $readmemh("x_in.hex", xq);
        $readmemh("y_ref.hex", yref);
    end

    prism_top #(.L(L), .P(P)) dut (
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

    // Per-state clock counts, so the cycle budget can be read off the run.
    integer cnt [0:16];
    integer s;
    initial for (s = 0; s < 17; s = s + 1) cnt[s] = 0;
    always @(posedge clk) if (run) cnt[dut.st] = cnt[dut.st] + 1;

    integer sn = 0;
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

        wait (done == 1'b1);
        repeat (4) @(posedge clk);

        bad = 0;
        for (i = 0; i < HP*P; i = i + 1)
            if (got[i] !== $signed(yref[i])) begin
                bad = bad + 1;
                if (bad < 6)
                    $display("MISMATCH i=%0d got=%0d want=%0d", i, got[i], $signed(yref[i]));
            end
        $display("cycles=%0d mismatches=%0d of %0d", cyc, bad, HP*P);
        $display("states ld=%0d sum=%0d d1=%0d d2=%0d d3=%0d d4=%0d q1=%0d q2=%0d inv=%0d fold=%0d in=%0d run=%0d out=%0d skip=%0d emit=%0d",
                 cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6], cnt[7], cnt[8], cnt[9],
                 cnt[10], cnt[11], cnt[12], cnt[13], cnt[14], cnt[15]);
        $finish;
    end
`ifdef TRACE_STOP
    always @(posedge clk) if ($time >= `TRACE_STOP) $finish;
`endif

endmodule
