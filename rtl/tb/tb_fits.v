// Drives fits_top through one forecast and compares against fits_ref.py.
`timescale 1ns / 1ps

module tb_fits;
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
        $readmemh("fits_x_in.hex", xq);
        $readmemh("fits_y_ref.hex", yref);
    end

    initial begin
        `ifdef TRACE
        $dumpfile("fits.vcd");
        $dumpvars(0, tb_fits);
        `endif
    end

    fits_top #(.L(L), .H(H)) dut (
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

        for (i = 0; i < 200000 && !done; i = i + 1) @(posedge clk);
        if (!done) begin
            $display("TIMEOUT st=%0d i=%0d k=%0d n=%0d cyc=%0d",
                     dut.st, dut.i, dut.k, dut.n, cyc);
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
