const std = @import("std");
const Allocator = std.mem.Allocator;

const assert = std.debug.assert;

const TokenKind = enum {
    lparen,
    rparen,
    lbrace,
    rbrace,
    comma,
    semi,
    dot,
    minus,
    plus,
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

    keyword_and,
    keyword_class,
    keyword_else,
    keyword_false,
    keyword_for,
    keyword_fun,
    keyword_if,
    keyword_nil,
    keyword_or,
    keyword_print,
    keyword_return,
    keyword_super,
    keyword_this,
    keyword_true,
    keyword_var,
    keyword_while,

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
                const unquoted = unquote(self.text, std.heap.page_allocator) catch unreachable;
                defer unquoted.deinit(std.heap.page_allocator);
                try writer.print("STRING {s} {s}", .{ self.text, unquoted.value });
                return;
            },
            .ident => {
                try writer.print("IDENTIFIER {s} null", .{self.text});
                return;
            },
            .number => {
                try writer.print("NUMBER {s} ", .{self.text});

                const parsed = std.fmt.parseFloat(f64, self.text) catch unreachable;
                try print_number(writer, parsed);
                return;
            },

            .keyword_and => "AND and null",
            .keyword_class => "CLASS class null",
            .keyword_else => "ELSE else null",
            .keyword_false => "FALSE false null",
            .keyword_for => "FOR for null",
            .keyword_fun => "FUN fun null",
            .keyword_if => "IF if null",
            .keyword_nil => "NIL nil null",
            .keyword_or => "OR or null",
            .keyword_print => "PRINT print null",
            .keyword_return => "RETURN return null",
            .keyword_super => "SUPER super null",
            .keyword_this => "THIS this null",
            .keyword_true => "TRUE true null",
            .keyword_var => "VAR var null",
            .keyword_while => "WHILE while null",

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

const keywords = std.StaticStringMap(TokenKind).initComptime(.{
    .{ "and", .keyword_and },
    .{ "class", .keyword_class },
    .{ "else", .keyword_else },
    .{ "false", .keyword_false },
    .{ "for", .keyword_for },
    .{ "fun", .keyword_fun },
    .{ "if", .keyword_if },
    .{ "nil", .keyword_nil },
    .{ "or", .keyword_or },
    .{ "print", .keyword_print },
    .{ "return", .keyword_return },
    .{ "super", .keyword_super },
    .{ "this", .keyword_this },
    .{ "true", .keyword_true },
    .{ "var", .keyword_var },
    .{ "while", .keyword_while },
});

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
        switch (self.current()) {
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
            'a'...'z', 'A'...'Z', '_' => try self.tokenizeIdentOrKeyword(),
            ' ', '\n', '\r', '\t' => try self.tokenizeWhitespace(),
            else => |c| {
                self.report_error("Unexpected character: {c}", .{c});
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
                self.report_error("Unterminated string.", .{});
                return;
            }
            if (self.current() == '\\') {
                self.advance();
                if (!self.hasNext()) {
                    self.report_error("Invalid escape sequence, unexpected EOF", .{});
                    return;
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

    fn tokenizeIdentOrKeyword(self: *Self) Allocator.Error!void {
        assert(std.ascii.isAlphabetic(self.current()) or self.current() == '_');
        const start = self.offset;
        self.skipIdent();

        const text = self.src.contents[start..self.offset];
        const tok = keywords.get(text) orelse .ident;
        try self.addTokenStartingAt(tok, start);
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
        while (self.hasNext()) {
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
        std.io.getStdErr().writer().print("[line {d}] Error: {s}\n", .{ position.line, msg }) catch unreachable;
        self.erroed = true;
    }
};

fn Cow(comptime T: type) type {
    return struct {
        is_owned: bool,
        value: T,

        const Self = @This();

        fn deinit(self: *const Self, allocator: Allocator) void {
            if (self.is_owned) {
                allocator.free(self.value);
            }
        }
    };
}

fn unquote(string: []const u8, allocator: Allocator) Allocator.Error!Cow([]const u8) {
    assert(string.len >= 2 and string[0] == '"' and string[string.len - 1] == '"');
    var naive_unquoted = string[1 .. string.len - 1];
    if (std.mem.indexOfScalar(u8, naive_unquoted, '\\')) |next_escape| {
        var list = std.ArrayList(u8).init(allocator);
        try list.appendSlice(naive_unquoted[0..next_escape]);
        var i = next_escape + 1;
        // If this fails, we must have had an invalid escape sequence, which should not be
        // possible at this point.
        assert(i <= naive_unquoted.len);
        while (std.mem.indexOfScalarPos(u8, naive_unquoted, i, '\\')) |esc| {
            try list.appendSlice(naive_unquoted[i..esc]);
            i = esc + 1;
            // See reasoning above
            assert(i <= naive_unquoted.len);
        }
        try list.appendSlice(naive_unquoted[i..]);
        return Cow([]const u8){ .is_owned = true, .value = list.items };
    } else {
        return Cow([]const u8){ .is_owned = false, .value = naive_unquoted };
    }
}

const Command = enum {
    tokenize,
    parse,

    pub fn parse(arg: []u8) error{UnknownCommand}!Command {
        if (std.mem.eql(u8, arg, "tokenize")) {
            return .tokenize;
        }
        if (std.mem.eql(u8, arg, "parse")) {
            return .parse;
        }
        return error.UnknownCommand;
    }
};

pub fn print_tokenize_result(src: *const Source, result: Lexer.Result) !void {
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

const Unary = enum {
    neg,
    not,
};

const Binary = enum {
    plus,
    minus,
    times,
    div,
    lt,
    leq,
    gt,
    geq,
    eq,
    neq,

    fn getPrecedence(self: Binary) u8 {
        return switch (self) {
            .lt, .leq, .gt, .geq, .eq, .neq => 1,
            .plus, .minus => 2,
            .times, .div => 3,
        };
    }
};

const Expr = union(enum) {
    bool: bool,
    nil,
    number: f64,
    string: []const u8,
    group: *Expr,
    unary_op: struct {
        op: Unary,
        operand: *Expr,
    },
    binary_op: struct {
        lhs: *Expr,
        op: Binary,
        rhs: *Expr,
    },

    pub fn format(
        self: Expr,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .nil => try writer.print("nil", .{}),
            .bool => |value| try writer.print("{}", .{value}),
            .number => |value| try print_number(writer, value),
            .string => |value| try writer.print("{s}", .{value}),
            .group => |inner| try writer.print("(group {})", .{inner}),
            .unary_op => |payload| {
                const op = switch (payload.op) {
                    .neg => "-",
                    .not => "!",
                };
                try writer.print("({s} {})", .{ op, payload.operand });
            },
            .binary_op => |payload| {
                const op = switch (payload.op) {
                    .plus => "+",
                    .minus => "-",
                    .times => "*",
                    .div => "/",
                    .lt => "<",
                    .leq => "<=",
                    .gt => ">",
                    .geq => ">=",
                    .eq => "==",
                    .neq => "!=",
                };
                try writer.print("({s} {} {})", .{ op, payload.lhs, payload.rhs });
            },
        }
    }
};

const Tree = struct {
    root: Expr,
};

const Parser = struct {
    src: *const Source,
    tokens: *const Tokens,
    allocator: Allocator,
    index: usize,

    const Error = error{
        ExpectedExpression,
        ExpectedBinaryOperator,
        UnexpectedToken,
        UnexpectedEof,
    } || Allocator.Error;

    const Self = @This();

    pub fn parse(src: *const Source, tokens: *const Tokens, allocator: Allocator) Error!*Expr {
        var self = Parser{
            .src = src,
            .tokens = tokens,
            .allocator = allocator,
            .index = 0,
        };
        self.skipCommentsAndWhitespace();

        return self.parseExpr(0);
    }

    fn parseExpr(self: *Self, min_prec: u8) Error!*Expr {
        var lhs = try self.parseAtom();
        while (true) {
            const op: Binary = switch (self.current()) {
                .minus => .minus,
                .plus => .plus,
                .star => .times,
                .slash => .div,
                .lt => .lt,
                .gt => .gt,
                .eq_eq => .eq,
                .not_eq => .neq,
                .leq => .leq,
                .geq => .geq,
                .eof, .rparen => break,
                .comment, .whitespace => unreachable,
                else => return error.ExpectedBinaryOperator,
            };

            const prec = op.getPrecedence();
            if (prec < min_prec) {
                break;
            }

            self.advance();
            const rhs = try self.parseExpr(prec + 1);
            const expr = try self.allocator.create(Expr);
            expr.* = Expr{ .binary_op = .{ .lhs = lhs, .op = op, .rhs = rhs } };
            lhs = expr;
        }

        return lhs;
    }

    fn parseAtom(self: *Self) Error!*Expr {
        while (true) {
            switch (self.current()) {
                .lparen => {
                    self.advance();
                    const inner = try self.parseExpr(0);
                    assert(self.current() == .rparen);
                    self.advance();
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .group = inner };
                    return expr;
                },
                // .rparen,
                // .lbrace,
                // .rbrace,
                // .comma,
                // .dot,
                .minus => {
                    self.advance();
                    const operand = try self.parseAtom();
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .unary_op = .{ .op = .neg, .operand = operand } };
                    return expr;
                },
                // .plus,
                // .semi,
                // .star,
                // .slash,
                // .lt,
                // .gt,
                .not => {
                    self.advance();
                    const operand = try self.parseAtom();
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .unary_op = .{ .op = .not, .operand = operand } };
                    return expr;
                },
                // .eq_eq,
                // .not_eq,
                // .leq,
                // .geq,
                // .assign,
                .string => {
                    defer self.advance();
                    // TODO: review me! Lifetimes might be confusing here
                    const unquoted = try unquote(self.currentText(), self.allocator);
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .string = unquoted.value };
                    return expr;
                },
                // .ident => {},
                .number => {
                    defer self.advance();
                    const parsed = std.fmt.parseFloat(f64, self.currentText()) catch @panic("should have already checked this");
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .number = parsed };
                    return expr;
                },

                // .keyword_and,
                // .keyword_class,
                // .keyword_else,
                // .keyword_for,
                // .keyword_fun,
                // .keyword_if,
                .keyword_nil => {
                    defer self.advance();
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr.nil;
                    return expr;
                },
                .keyword_false => {
                    defer self.advance();
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .bool = false };
                    return expr;
                },
                .keyword_true => {
                    defer self.advance();
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .bool = true };
                    return expr;
                },
                // .keyword_or,
                // .keyword_print,
                // .keyword_return,
                // .keyword_super,
                // .keyword_this,
                // .keyword_var,
                // .keyword_while,

                .comment, .whitespace => unreachable,
                .eof => return error.UnexpectedEof,

                else => return error.UnexpectedToken,
            }
        }
    }

    fn advance(self: *Self) void {
        self.index += 1;
        assert(self.index <= self.tokens.len);
        self.skipCommentsAndWhitespace();
    }

    fn skipCommentsAndWhitespace(self: *Self) void {
        while (true) {
            const tok = self.current();
            if (tok != .comment and tok != .whitespace) {
                break;
            }
            self.index += 1;
            assert(self.index <= self.tokens.len);
        }
    }

    fn current(self: *const Self) TokenKind {
        return self.tokens.items(.kind)[self.index];
    }

    fn currentText(self: *const Self) []const u8 {
        assert(self.index + 1 < self.tokens.len);
        const offsets = self.tokens.items(.offset);
        const start = offsets[self.index];
        const end = offsets[self.index + 1];
        return self.src.contents[start..end];
    }

    fn hasNext(self: *const Self) bool {
        return self.index < self.tokens.len;
    }
};

pub fn print_parse_result(src: *const Source, result: Lexer.Result, allocator: Allocator) !void {
    var stdout = std.io.getStdOut().writer();
    if (result.erroed) {
        std.process.exit(65);
        return;
    }
    const expr = Parser.parse(src, &result.tokens, allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            // TODO: implement nice error messages
            std.process.exit(65);
        },
    };
    try stdout.print("{s}\n", .{expr});
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var alloc = arena.allocator();

    // You can use print statements as follows for debugging, they'll be visible when running tests.
    std.debug.print("Logs from your program will appear here!\n", .{});

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    if (args.len < 3) {
        std.debug.print("Usage: ./your_program.sh tokenize <filename>\n", .{});
        std.process.exit(1);
    }

    const filename = args[2];

    const command = Command.parse(args[1]) catch {
        std.debug.print("Unknown command: {s}\n", .{args[1]});
        std.process.exit(1);
    };

    const src = try Source.load(filename, alloc);
    defer alloc.free(src.contents);

    const lex_result = try Lexer.lex(&src, alloc);

    switch (command) {
        .tokenize => try print_tokenize_result(&src, lex_result),
        .parse => try print_parse_result(&src, lex_result, alloc),
    }
}

fn print_number(writer: anytype, number: f64) !void {
    // Little hack to make sure we print the '.0' at the end, even if it is an integral value.
    // 4k should be enough, right?
    var buf: [1 << 12]u8 = undefined;
    const formatted = std.fmt.bufPrint(&buf, "{d}", .{number}) catch unreachable;
    const hasDot = std.mem.indexOfScalar(u8, formatted, '.') != null;
    if (!hasDot) {
        try writer.print("{d:.1}", .{number});
    } else {
        try writer.print("{d}", .{number});
    }
}
