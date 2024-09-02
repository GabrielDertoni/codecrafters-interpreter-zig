const std = @import("std");
const Allocator = std.mem.Allocator;

const assert = std.debug.assert;

const TokenKind = enum {
    lparen,
    rparen,
    lbrace,
    rbrace,
    comma,
    dot,
    minus,
    plus,
    semi,
    star,
    slash,
    lt,
    gt,
    not,
    eq_eq,
    not_eq,
    leq,
    geq,
    assign,
    string,
    ident,
    number,

    whitespace,
    comment,
    eof,
};

const Token = struct {
    kind: TokenKind,
    offset: u32,
};

const TokenSpan = struct {
    kind: TokenKind,
    text: []const u8,

    pub fn format(
        self: TokenSpan,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        const str = switch (self.kind) {
            .lparen => "LEFT_PAREN ( null",
            .rparen => "RIGHT_PAREN ) null",
            .lbrace => "LEFT_BRACE { null",
            .rbrace => "RIGHT_BRACE } null",
            .comma => "COMMA , null",
            .dot => "DOT . null",
            .minus => "MINUS - null",
            .plus => "PLUS + null",
            .semi => "SEMICOLON ; null",
            .star => "STAR * null",
            .slash => "SLASH / null",
            .lt => "LESS < null",
            .gt => "GREATER > null",
            .not => "BANG ! null",
            .eq_eq => "EQUAL_EQUAL == null",
            .not_eq => "BANG_EQUAL != null",
            .leq => "LESS_EQUAL <= null",
            .geq => "GREATER_EQUAL >= null",
            .assign => "EQUAL = null",
            .string => {
                try writer.print("STRING {s} null", .{self.text});
                return;
            },
            .ident => {
                try writer.print("IDENTIFIER {s} null", .{self.text});
                return;
            },
            .number => {
                try writer.print("NUMBER {s} ", .{self.text});

                const parsed = std.fmt.parseFloat(f64, self.text) catch unreachable;
                // Little hack to make sure we print the '.0' at the end, even if it is an integral value.
                // 4k should be enough, right?
                var buf: [1 << 12]u8 = undefined;
                const formatted = std.fmt.bufPrint(&buf, "{d}", .{parsed}) catch unreachable;
                const hasDot = std.mem.indexOfScalar(u8, formatted, '.') != null;
                if (!hasDot) {
                    try writer.print("{d:.1}", .{parsed});
                } else {
                    try writer.print("{d}", .{parsed});
                }
                return;
            },

            .whitespace => "WHITESPACE   null",
            .comment => "COMMENT // null",
            .eof => "EOF  null",
        };

        try writer.print("{s}", .{str});
    }
};

const Tokens = std.MultiArrayList(Token);

// Lifetime of the program
const Source = struct {
    contents: []const u8,
    fname: []const u8,

    const Self = @This();

    pub fn load(fname: []const u8, allocator: Allocator) !Source {
        const contents = try std.fs.cwd().readFileAlloc(allocator, fname, std.math.maxInt(u32));
        return Source{
            .contents = contents,
            .fname = fname,
        };
    }

    pub fn unload(self: *Self, allocator: Allocator) void {
        allocator.free(self.contents);
    }

    pub fn computePositionFromOffset(self: *const Self, offset: u32) Position {
        assert(offset <= self.contents.len);
        var line: u32 = 1;
        var column: u32 = 1;
        var i: usize = 0;
        while (i < offset) {
            const c = self.contents[i];
            if (c == '\r' or c == '\n') {
                if (c == '\r' and i + 1 < self.contents.len and self.contents[i + 1] == '\n') {
                    i += 1;
                }
                line += 1;
                column = 0;
            }
            column += 1;
            i += 1;
        }

        return Position{
            .line = line,
            .column = column,
        };
    }
};

const Position = struct {
    line: u32,
    column: u32,
};

const Lexer = struct {
    src: *const Source,
    offset: u32,
    tokens: Tokens,
    allocator: Allocator,
    scratch: Allocator,
    erroed: bool,

    const Self = @This();

    const Result = struct {
        tokens: Tokens,
        erroed: bool,
    };

    pub fn lex(src: *const Source, allocator: Allocator) Allocator.Error!Result {
        var scratch = std.heap.stackFallback(1 << 13, std.heap.page_allocator);
        var self = Self{
            .src = src,
            .offset = 0,
            .tokens = Tokens{},
            .allocator = allocator,
            .scratch = scratch.get(),
            .erroed = false,
        };

        while (self.offset < self.src.contents.len) {
            try self.nextToken();
        }
        try self.addToken(.eof);

        return Result{
            .tokens = self.tokens,
            .erroed = self.erroed,
        };
    }

    fn nextToken(self: *Self) Allocator.Error!void {
        const c = self.current();
        switch (c) {
            '(' => try self.tokenizeSingle(.lparen),
            ')' => try self.tokenizeSingle(.rparen),
            '{' => try self.tokenizeSingle(.lbrace),
            '}' => try self.tokenizeSingle(.rbrace),
            ',' => try self.tokenizeSingle(.comma),
            '.' => try self.tokenizeSingle(.dot),
            '-' => try self.tokenizeSingle(.minus),
            '+' => try self.tokenizeSingle(.plus),
            ';' => try self.tokenizeSingle(.semi),
            '*' => try self.tokenizeSingle(.star),
            '/' => {
                if (self.peek() == '/') {
                    try self.tokenizeLineComment();
                } else {
                    try self.tokenizeSingle(.slash);
                }
            },
            '<' => {
                if (self.peek() == '=') {
                    try self.tokenizeN(.leq, 2);
                } else {
                    try self.tokenizeSingle(.lt);
                }
            },
            '>' => {
                if (self.peek() == '=') {
                    try self.tokenizeN(.geq, 2);
                } else {
                    try self.tokenizeSingle(.gt);
                }
            },
            '=' => {
                if (self.peek() == '=') {
                    try self.tokenizeN(.eq_eq, 2);
                } else {
                    try self.tokenizeSingle(.assign);
                }
            },
            '!' => {
                if (self.peek() == '=') {
                    try self.tokenizeN(.not_eq, 2);
                } else {
                    try self.tokenizeSingle(.not);
                }
            },
            '"' => try self.tokenizeString(),
            '0'...'9' => try self.tokenizeNumber(),
            'a'...'z', 'A'...'Z', '_' => try self.tokenizeIdent(),
            ' ', '\n', '\r', '\t' => try self.tokenizeWhitespace(),
            else => {
                self.report_error("Error: Unexpected character: {c}", .{c});
                self.advance();
            },
        }
    }

    fn addToken(self: *Self, tok: TokenKind) Allocator.Error!void {
        try self.addTokenStartingAt(tok, self.offset);
    }

    fn addTokenStartingAt(self: *Self, tok: TokenKind, offset: u32) Allocator.Error!void {
        try self.tokens.append(self.allocator, Token{
            .kind = tok,
            .offset = offset,
        });
    }

    fn current(self: *const Self) u8 {
        assert(self.offset < self.src.contents.len);
        return self.src.contents[self.offset];
    }

    fn peek(self: *const Self) ?u8 {
        return self.peekN(1);
    }

    fn peekN(self: *const Self, n: u32) ?u8 {
        return if (self.offset + n < self.src.contents.len)
            self.src.contents[self.offset + n]
        else
            null;
    }

    fn hasNext(self: *const Self) bool {
        return self.offset < self.src.contents.len;
    }

    fn advance(self: *Self) void {
        self.advanceBy(1);
    }

    fn advanceBy(self: *Self, n: u32) void {
        self.offset += n;
        assert(self.offset <= self.src.contents.len);
    }

    fn tokenizeSingle(self: *Self, tok: TokenKind) Allocator.Error!void {
        try self.addToken(tok);
        self.advance();
    }

    fn tokenizeN(self: *Self, tok: TokenKind, size: u32) Allocator.Error!void {
        try self.addToken(tok);
        self.advanceBy(size);
    }

    fn tokenizeString(self: *Self) Allocator.Error!void {
        assert(self.current() == '"');
        const start = self.offset;
        self.advance(); // Skip open quote
        while (true) {
            if (!self.hasNext()) {
                self.report_error("Unexpected EOF", .{});
                break;
            }
            if (self.current() == '\\') {
                self.advance();
                if (!self.hasNext()) {
                    self.report_error("Invalid escape sequence, unexpected EOF", .{});
                    break;
                }
            } else if (self.current() == '"') {
                self.advance();
                break;
            }
            self.advance();
        }
        try self.addTokenStartingAt(.string, start);
    }

    fn tokenizeNumber(self: *Self) Allocator.Error!void {
        assert(std.ascii.isDigit(self.current()));
        const start = self.offset;
        self.skipNumber();
        try self.addTokenStartingAt(.number, start);
    }

    fn tokenizeIdent(self: *Self) Allocator.Error!void {
        assert(std.ascii.isAlphabetic(self.current()) or self.current() == '_');
        const start = self.offset;
        self.skipIdent();
        try self.addTokenStartingAt(.ident, start);
    }

    fn tokenizeWhitespace(self: *Self) Allocator.Error!void {
        const start = self.offset;
        self.skipWhitespace();
        try self.addTokenStartingAt(.whitespace, start);
    }

    fn tokenizeLineComment(self: *Self) Allocator.Error!void {
        const start = self.offset;
        self.skipLineComment();
        try self.addTokenStartingAt(.comment, start);
    }

    fn skipWhitespace(self: *Self) void {
        while (self.hasNext() and std.ascii.isWhitespace(self.current())) {
            self.advance();
        }
    }

    fn skipLineComment(self: *Self) void {
        while (self.hasNext()) {
            switch (self.current()) {
                '\n', '\r' => {
                    self.skipNewLine();
                    break;
                },
                else => self.advance(),
            }
        }
    }

    fn skipNumber(self: *Self) void {
        // Integral part
        while (self.hasNext() and std.ascii.isDigit(self.current())) {
            self.advance();
        }
        if (self.hasNext() and self.current() == '.') {
            self.advance();
            // Fractional part
            while (self.hasNext() and std.ascii.isDigit(self.current())) {
                self.advance();
            }
        }
    }

    fn skipIdent(self: *Self) void {
        if (!(std.ascii.isAlphabetic(self.current()) or self.current() == '_')) {
            return;
        }
        self.advance();
        while (true) {
            switch (self.current()) {
                'a'...'z', 'A'...'Z', '0'...'9', '_' => self.advance(),
                else => break,
            }
        }
    }

    fn skipNewLine(self: *Self) void {
        if (self.current() == '\r') {
            self.advance();
            if (self.current() == '\n') {
                self.advance();
            }
        } else if (self.current() == '\n') {
            self.advance();
        }
    }

    fn report_error(self: *Self, comptime fmt: []const u8, args: anytype) void {
        // FIXME: remove from here
        const position = self.src.computePositionFromOffset(self.offset);
        const msg = std.fmt.allocPrint(self.scratch, fmt, args) catch unreachable;
        std.io.getStdErr().writer().print("[line {d}] {s}\n", .{ position.line, msg }) catch unreachable;
        self.erroed = true;
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    var alloc = arena.allocator();

    // You can use print statements as follows for debugging, they'll be visible when running tests.
    std.debug.print("Logs from your program will appear here!\n", .{});

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    if (args.len < 3) {
        std.debug.print("Usage: ./your_program.sh tokenize <filename>\n", .{});
        std.process.exit(1);
    }

    const command = args[1];
    const filename = args[2];

    if (!std.mem.eql(u8, command, "tokenize")) {
        std.debug.print("Unknown command: {s}\n", .{command});
        std.process.exit(1);
    }

    const src = try Source.load(filename, alloc);
    defer alloc.free(src.contents);

    const result = try Lexer.lex(&src, alloc);

    var stdout = std.io.getStdOut().writer();

    for (0..result.tokens.len) |i| {
        const tok = result.tokens.get(i);
        if (tok.kind == .whitespace or tok.kind == .comment) {
            continue;
        }
        if (tok.kind == .eof) {
            break;
        }
        const end = result.tokens.items(.offset)[i + 1];
        const tokenSpan = TokenSpan{
            .kind = tok.kind,
            .text = src.contents[tok.offset..end],
        };
        try stdout.print("{s}\n", .{tokenSpan});
    }

    try stdout.print("EOF  null\n", .{}); // Placeholder, remove this line when implementing the scanner
    if (result.erroed) {
        std.process.exit(65);
    }
}
