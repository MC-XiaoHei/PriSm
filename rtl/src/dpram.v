// One write port, one registered read port, described so that the same source
// infers block RAM on every vendor. Vendor synthesis reports confirm the
// mapping; nothing here is tool specific.
`timescale 1ns / 1ps

module dpram #(
    parameter integer WIDTH = 16,
    parameter integer DEPTH = 1024,
    parameter integer AW    = 10
)(
    input  wire                 clk,
    input  wire                 we,
    input  wire [AW-1:0]        waddr,
    input  wire [WIDTH-1:0]     wdata,
    input  wire [AW-1:0]        raddr,
    output wire [WIDTH-1:0]     rdata
);

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [WIDTH-1:0] q;

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        q <= mem[raddr];
    end

    assign rdata = q;

endmodule
