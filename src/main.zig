const std = @import("std");
const Allocator = std.mem.Allocator;

const assert = std.debug.assert;

fn FmtStr(comptime T: type) type {
    return struct {
        value: T,

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = options;
            _ = fmt;
            try writer.print("{s}", .{self.value});
        }
    };
}

fn fmtStr(value: anytype) FmtStr(@TypeOf(value)) {
    return FmtStr(@TypeOf(value)){ .value = value };
}

fn FmtFields(comptime T: type) type {
    return struct {
        value: T,

        pub fn format(
            self: @This(),
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;

            const info = @typeInfo(T);
            const new_fmt = comptime switch (info) {
                .Struct => |struct_info| blk: {
                    var f: []const u8 = "{{";
                    for (0.., struct_info.fields) |i, field| {
                        f = f ++ "." ++ field.name ++ " = {}";
                        if (i != struct_info.fields.len - 1) {
                            f = f ++ ", ";
                        }
                    }
                    f = f ++ "}}";
                    break :blk f;
                },
                else => @compileError("fmtFields can only work with struct"),
            };
            try writer.print(new_fmt, self.value);
        }
    };
}

fn fmtFields(value: anytype) FmtFields(@TypeOf(value)) {
    return FmtFields(@TypeOf(value)){ .value = value };
}

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

    pub fn loadStdin(allocator: Allocator) !Source {
        var stdin = std.io.getStdIn().reader();
        const contents = try stdin.readAllAlloc(allocator, std.math.maxInt(u32));
        return Source{
            .contents = contents,
            .fname = "stdin",
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
            .fname = self.fname,
            .line = line,
            .column = column,
        };
    }
};

const Position = struct {
    fname: []const u8,
    line: u32,
    column: u32,

    pub fn format(
        self: Position,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}:{}:{}", .{ self.fname, self.line, self.column });
    }
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
    evaluate,
    run,

    pub fn parse(arg: []u8) error{UnknownCommand}!Command {
        if (std.mem.eql(u8, arg, "tokenize")) {
            return .tokenize;
        }
        if (std.mem.eql(u8, arg, "parse")) {
            return .parse;
        }
        if (std.mem.eql(u8, arg, "evaluate")) {
            return .evaluate;
        }
        if (std.mem.eql(u8, arg, "run")) {
            return .run;
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

const Precedence = struct {
    left_assoc: u8,
    right_assoc: u8,
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

    fn getPrecedence(self: Binary) Precedence {
        return switch (self) {
            .lt, .leq, .gt, .geq, .eq, .neq => .{ .left_assoc = 4, .right_assoc = 5 },
            .plus, .minus => .{ .left_assoc = 6, .right_assoc = 7 },
            .times, .div => .{ .left_assoc = 8, .right_assoc = 9 },
        };
    }
};

const Stmt = union(enum) {
    print: *Expr,
    expr: *Expr,
    var_decl: struct {
        name: Symbol,
        value: ?*Expr,
    },
    block: Stmts,
};

const Symbol = Interner.Index; // interned

const Stmts = std.ArrayList(Stmt);

const Expr = union(enum) {
    nil,
    bool: bool,
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
    var_ref: Symbol,
    assign: struct {
        target: Symbol,
        value: *Expr,
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
            .var_ref => |index| {
                try writer.print("{s}", .{g_interner.get(index).?});
            },
            .assign => |payload| {
                try writer.print("(= {s} {})", .{ g_interner.get(payload.target).?, payload.value });
            },
        }
    }
};

const Interner = struct {
    symbol_hash: std.StringHashMap(Index),
    symbols: std.ArrayList([]const u8),

    const Self = @This();
    const Index = u32;

    pub fn init(allocator: Allocator) Self {
        return Self{
            .symbol_hash = std.StringHashMap(Symbol).init(allocator),
            .symbols = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: Self) void {
        self.symbol_hash.deinit();
        self.symbols.deinit();
    }

    pub fn internOrGet(self: *Self, string: []const u8) Allocator.Error!Index {
        const result = try self.symbol_hash.getOrPut(string);
        if (!result.found_existing) {
            const index: Index = @intCast(self.symbols.items.len);
            result.value_ptr.* = index;
            try self.symbols.append(try self.symbols.allocator.dupe(u8, string));
            return index;
        }
        return result.value_ptr.*;
    }

    pub fn get(self: *const Self, index: Index) ?[]const u8 {
        return if (index < self.symbols.items.len) self.symbols.items[index] else null;
    }
};

var g_interner: Interner = undefined;

const Parser = struct {
    src: *const Source,
    tokens: *const Tokens,
    allocator: Allocator,
    interner: *Interner,
    index: usize,

    const Error = error{
        ExpectedExpression,
        ExpectedBinaryOperator,
        ExpectedSemi,
        ExpectedIdent,
        ExpectedAssign,
        ExpectedRParen,
        ExpectedRBrace,
        ExpressionCannotBeAssigned,
        UnexpectedToken,
        UnexpectedEof,
    } || Allocator.Error;

    const Self = @This();

    pub fn parse(src: *const Source, tokens: *const Tokens, allocator: Allocator) Error!Stmts {
        var self = Parser{
            .src = src,
            .tokens = tokens,
            .allocator = allocator,
            .interner = &g_interner,
            .index = 0,
        };
        self.skipCommentsAndWhitespace();
        return self.parseStmts();
    }

    pub fn parseSingleExpr(src: *const Source, tokens: *const Tokens, allocator: Allocator) Error!*Expr {
        var self = Parser{
            .src = src,
            .tokens = tokens,
            .allocator = allocator,
            .interner = &g_interner,
            .index = 0,
        };
        self.skipCommentsAndWhitespace();
        return self.parseExpr(0);
    }

    fn parseStmts(self: *Self) Error!Stmts {
        // TODO: should probably use a different allocator? This may cause way too much fragmentation
        // because of vector resizes.
        var stmts = Stmts.init(self.allocator);
        while (self.hasNext()) {
            const tok = self.current();
            if (tok == .eof or tok == .rbrace) {
                break;
            }
            const stmt = try self.parseStmt();
            try stmts.append(stmt);
        }
        return stmts;
    }

    fn parseStmt(self: *Self) Error!Stmt {
        switch (self.current()) {
            .keyword_print => {
                self.advance();
                const expr = try self.parseExpr(0);
                if (self.current() != .semi) {
                    return error.ExpectedSemi;
                }
                self.advance();
                return Stmt{ .print = expr };
            },
            .keyword_var => {
                self.advance();
                const name = try self.expectIdent();
                if (self.current() == .semi) {
                    self.advance();
                    // No assignment
                    return Stmt{ .var_decl = .{ .name = name, .value = null } };
                }
                if (self.current() != .assign) {
                    return error.ExpectedAssign;
                }
                self.advance();
                const value = try self.parseExpr(0);
                if (self.current() != .semi) {
                    return error.ExpectedSemi;
                }
                self.advance();
                return Stmt{ .var_decl = .{ .name = name, .value = value } };
            },
            .lbrace => {
                self.advance();
                const stmts = try self.parseStmts();
                if (self.current() != .rbrace) {
                    return error.ExpectedRBrace;
                }
                self.advance();
                return Stmt{ .block = stmts };
            },
            else => {
                const expr = try self.parseExpr(0);
                if (self.current() != .semi) {
                    return error.ExpectedSemi;
                }
                self.advance();
                return Stmt{ .expr = expr };
            },
        }
    }

    fn expectIdent(self: *Self) Error!Symbol {
        if (self.current() != .ident) {
            return error.ExpectedIdent;
        }
        const text = self.currentText();
        const ident = self.interner.internOrGet(text);
        self.advance();
        return ident;
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
                .assign => {
                    // NOTE: this is a bit unfortunate. The way we're doing this it's just a lot easier to parse the lhs as
                    // a full expression an then check it is what we expect. In the future we might be able to do something
                    // smarter though.
                    self.advance();
                    defer self.allocator.destroy(lhs);
                    const sym = switch (lhs.*) {
                        .var_ref => |sym| sym,
                        else => {
                            std.log.err("expression cannot be assigned: {}", .{fmtFields(.{
                                .target = lhs.*,
                                .position = self.src.computePositionFromOffset(self.currentOffset()),
                            })});
                            return error.ExpressionCannotBeAssigned;
                        },
                    };
                    const value = try self.parseExpr(0);
                    // Lets just reuse the allocation!
                    lhs.* = Expr{ .assign = .{ .target = sym, .value = value } };
                    continue;
                },
                .eof, .rparen, .semi => break,
                .comment, .whitespace => unreachable,
                else => |tok| {
                    std.log.err("expected a binary operator: {}", .{fmtFields(.{
                        .token = tok,
                        .position = self.src.computePositionFromOffset(self.currentOffset()),
                    })});
                    return error.ExpectedBinaryOperator;
                },
            };

            const prec = op.getPrecedence();
            if (prec.left_assoc < min_prec) {
                break;
            }

            self.advance();
            const rhs = try self.parseExpr(prec.right_assoc);
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
                    if (self.current() != .rparen) {
                        return error.ExpectedRParen;
                    }
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
                .number => {
                    defer self.advance();
                    const parsed = std.fmt.parseFloat(f64, self.currentText()) catch @panic("should have already checked this");
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .number = parsed };
                    return expr;
                },
                .ident => {
                    defer self.advance();
                    const text = self.currentText();
                    const sym = try self.interner.internOrGet(text);
                    const expr = try self.allocator.create(Expr);
                    expr.* = Expr{ .var_ref = sym };
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

                else => |tok| {
                    std.log.err("unexpected token: {}", .{fmtFields(.{
                        .token = tok,
                        .position = self.src.computePositionFromOffset(self.currentOffset()),
                    })});
                    return error.UnexpectedToken;
                },
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

    fn currentOffset(self: *const Self) u32 {
        return self.tokens.items(.offset)[self.index];
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

const Utf8String = std.ArrayList(u8);
const Utf8StringSlice = []const u8;

const RuntimeError = error{
    ValueError,
    IoError,
    // VariableAlreadyDeclared,
    VariableIsNotDeclared,
} || Allocator.Error;

const Value = union(enum) {
    number: f64,
    string: Utf8String,
    bool: bool,
    nil,

    const Self = @This();

    fn assertNumber(self: Self) error{ValueError}!f64 {
        return switch (self) {
            .number => |value| value,
            else => error.ValueError,
        };
    }

    fn assertBool(self: Self) error{ValueError}!bool {
        return switch (self) {
            .bool => |value| value,
            else => error.ValueError,
        };
    }

    fn assertString(self: Self) error{ValueError}!Utf8StringSlice {
        return switch (self) {
            .string => |value| value.items,
            else => error.ValueError,
        };
    }

    // TODO: This should probably be garbage collected or ref-counted in the future
    pub fn clone(self: *const Self) Allocator.Error!Self {
        switch (self.*) {
            .string => |value| {
                var copy = Utf8String.init(value.allocator);
                try copy.appendSlice(value.items);
                return Self{ .string = copy };
            },
            // Memcpy will suffice.
            .number, .bool, .nil => return self.*,
        }
    }

    pub fn deinit(self: Self) void {
        switch (self) {
            .string => |value| value.deinit(),
            .number, .bool, .nil => {},
        }
    }

    pub fn format(
        self: Value,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            // .number => |value| try print_number(writer, value),
            .number => |value| try writer.print("{d}", .{value}),
            .string => |value| try writer.print("{s}", .{value.items}),
            .bool => |value| try writer.print("{}", .{value}),
            .nil => try writer.print("nil", .{}),
        }
    }
};

const Env = struct {
    variables: std.AutoHashMap(Symbol, Value),
    super: ?*Env,

    const Self = @This();

    pub fn decl(self: *Self, sym: Symbol, value: Value) RuntimeError!void {
        const result = try self.variables.getOrPut(sym);
        // if (result.found_existing) {
        //     return error.VariableAlreadyDeclared;
        // }
        result.value_ptr.* = value;
    }

    pub fn lookup(self: *const Self, sym: Symbol) RuntimeError!Value {
        const value_ptr = try self.lookupPtr(sym);
        return value_ptr.clone();
    }

    pub fn lookupPtr(self: Self, sym: Symbol) RuntimeError!*Value {
        var this = self;
        var next: ?*Env = &this;
        return while (next) |curr| {
            if (curr.variables.getPtr(sym)) |value| break value;
            next = curr.super;
        } else error.VariableIsNotDeclared;
    }

    pub fn set(self: *Self, sym: Symbol, value: Value) RuntimeError!void {
        const value_ptr = try self.lookupPtr(sym);
        value_ptr.* = value;
    }
};

const Evaluator = struct {
    env: Env,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .env = Env{
                .variables = std.AutoHashMap(Symbol, Value).init(allocator),
                .super = null,
            },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: Self) void {
        var this = self;
        this.env.variables.deinit();
    }

    pub fn run(self: *Self, stmts: Stmts.Slice) RuntimeError!void {
        var stdout = std.io.getStdOut().writer();
        for (stmts) |stmt| {
            switch (stmt) {
                .print => |expr| {
                    const value = try self.evaluate(expr);
                    stdout.print("{}\n", .{value}) catch return error.IoError;
                },
                .expr => |expr| {
                    _ = try self.evaluate(expr);
                },
                .var_decl => |payload| {
                    const value = if (payload.value) |expr|
                        try self.evaluate(expr)
                    else
                        Value.nil;
                    try self.env.decl(payload.name, value);
                },
                .block => |blk_stmts| {
                    var subEval = Evaluator{
                        .env = Env{
                            .variables = std.AutoHashMap(Symbol, Value).init(self.allocator),
                            .super = &self.env,
                        },
                        .allocator = self.allocator,
                    };
                    defer subEval.env.variables.deinit();
                    try subEval.run(blk_stmts.items);
                },
            }
        }
    }

    pub fn evaluate(self: *Self, expr: *const Expr) RuntimeError!Value {
        return switch (expr.*) {
            .nil => Value.nil,
            .bool => |value| Value{ .bool = value },
            .number => |value| Value{ .number = value },
            .string => |value| blk: {
                var string = Utf8String.init(self.allocator);
                try string.appendSlice(value);
                break :blk Value{ .string = string };
            },
            .group => |inner| self.evaluate(inner),
            .unary_op => |payload| blk: {
                const inner = try self.evaluate(payload.operand);
                defer inner.deinit();
                break :blk switch (payload.op) {
                    .neg => Value{ .number = -(try inner.assertNumber()) },
                    .not => Value{
                        .bool = switch (inner) {
                            .nil => true, // nil is falsy
                            .bool => |value| !value,
                            else => false, // other stuff is truthy
                        },
                    },
                };
            },
            .binary_op => |payload| blk: {
                const lhs = try self.evaluate(payload.lhs);
                defer lhs.deinit();
                const rhs = try self.evaluate(payload.rhs);
                defer rhs.deinit();
                break :blk switch (payload.op) {
                    .plus => switch (lhs) {
                        .number => |lhs_value| Value{ .number = lhs_value + (try rhs.assertNumber()) },
                        .string => |lhs_value| inner_blk: {
                            var result = Utf8String.init(self.allocator);
                            try result.appendSlice(lhs_value.items);
                            try result.appendSlice(try rhs.assertString());
                            break :inner_blk Value{ .string = result };
                        },
                        else => return error.ValueError,
                    },
                    .minus => Value{ .number = (try lhs.assertNumber()) - (try rhs.assertNumber()) },
                    .times => Value{ .number = (try lhs.assertNumber()) * (try rhs.assertNumber()) },
                    .div => Value{ .number = (try lhs.assertNumber()) / (try rhs.assertNumber()) },
                    .lt => Value{ .bool = (try lhs.assertNumber()) < (try rhs.assertNumber()) },
                    .leq => Value{ .bool = (try lhs.assertNumber()) <= (try rhs.assertNumber()) },
                    .gt => Value{ .bool = (try lhs.assertNumber()) > (try rhs.assertNumber()) },
                    .geq => Value{ .bool = (try lhs.assertNumber()) >= (try rhs.assertNumber()) },
                    // .lt => switch (lhs) {
                    //     .number => |lhs_value| Value{ .bool = lhs_value < (try rhs.assertNumber()) },
                    //     .bool => |lhs_value| Value{ .bool = compareBool(lhs_value, try rhs.assertBool()).compare(.lt) },
                    //     .string => |lhs_value| Value{ .bool = std.mem.order(u8, lhs_value.items, try rhs.assertString()) == .lt },
                    //     .nil => return error.ValueError,
                    // },
                    // .leq => switch (lhs) {
                    //     .number => |lhs_value| Value{ .bool = lhs_value <= (try rhs.assertNumber()) },
                    //     .bool => |lhs_value| Value{ .bool = compareBool(lhs_value, try rhs.assertBool()).compare(.lte) },
                    //     .string => |lhs_value| Value{ .bool = std.mem.order(u8, lhs_value.items, try rhs.assertString()).compare(.lte) },
                    //     .nil => return error.ValueError,
                    // },
                    // .gt => switch (lhs) {
                    //     .number => |lhs_value| Value{ .bool = lhs_value > (try rhs.assertNumber()) },
                    //     .bool => |lhs_value| Value{ .bool = compareBool(lhs_value, try rhs.assertBool()).compare(.gt) },
                    //     .string => |lhs_value| Value{ .bool = std.mem.order(u8, lhs_value.items, try rhs.assertString()).compare(.gt) },
                    //     .nil => return error.ValueError,
                    // },
                    // .geq => switch (lhs) {
                    //     .number => |lhs_value| Value{ .bool = lhs_value >= (try rhs.assertNumber()) },
                    //     .bool => |lhs_value| Value{ .bool = compareBool(lhs_value, try rhs.assertBool()).compare(.gte) },
                    //     .string => |lhs_value| Value{ .bool = std.mem.order(u8, lhs_value.items, try rhs.assertString()).compare(.gte) },
                    //     .nil => return error.ValueError,
                    // },
                    .eq => Value{ .bool = try evaluateEq(lhs, rhs) },
                    .neq => Value{ .bool = !try evaluateEq(lhs, rhs) },
                };
            },
            .var_ref => |sym| try self.env.lookup(sym),
            .assign => |payload| blk: {
                const value = try self.evaluate(payload.value);
                try self.env.set(payload.target, try value.clone());
                break :blk value;
            },
        };
    }
};

fn evaluateEq(lhs: Value, rhs: Value) RuntimeError!bool {
    return switch (lhs) {
        .number => |lhs_value| switch (rhs) {
            .number => |rhs_value| std.math.approxEqAbs(f64, lhs_value, rhs_value, 1e-12),
            else => false,
        },
        .bool => |lhs_value| switch (rhs) {
            .bool => |rhs_value| lhs_value == rhs_value,
            else => false,
        },
        .string => |lhs_value| switch (rhs) {
            .string => |rhs_value| std.mem.eql(u8, lhs_value.items, rhs_value.items),
            else => false,
        },
        .nil => return error.ValueError,
    };
}

fn compareBool(lhs: bool, rhs: bool) std.math.Order {
    return if (lhs)
        if (rhs) .eq else .gt
    else if (rhs) .lt else .eq;
}

pub fn main() !void {
    // Initialize stuff
    g_interner = Interner.init(std.heap.page_allocator);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var alloc = arena.allocator();

    // You can use print statements as follows for debugging, they'll be visible when running tests.
    std.debug.print("Logs from your program will appear here!\n", .{});

    var stdout = std.io.getStdOut().writer();
    var stderr = std.io.getStdErr().writer();

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

    const src = if (std.mem.eql(u8, filename, "-"))
        try Source.loadStdin(alloc)
    else
        try Source.load(filename, alloc);

    defer alloc.free(src.contents);

    const lex_result = try Lexer.lex(&src, alloc);
    if (command == .tokenize) {
        try print_tokenize_result(&src, lex_result);
        return;
    }

    if (lex_result.erroed) {
        std.process.exit(65);
        return;
    }

    const tokens = lex_result.tokens;

    if (command == .parse or command == .evaluate) {
        const expr = Parser.parseSingleExpr(&src, &tokens, alloc) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                stderr.print("{any}\n", .{err}) catch {};
                // TODO: implement nice error messages
                std.process.exit(65);
            },
        };

        if (command == .parse) {
            try stdout.print("{s}\n", .{expr});
        } else {
            var env = Evaluator.init(alloc);
            defer env.deinit();
            const value = env.evaluate(expr) catch |err| switch (err) {
                error.ValueError => {
                    stderr.print("ValueError\n", .{}) catch {};
                    std.process.exit(70);
                },
                else => return err,
            };
            try stdout.print("{}", .{value});
        }
        return;
    }

    const stmts = Parser.parse(&src, &tokens, alloc) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            stderr.print("{any}\n", .{err}) catch {};
            // TODO: implement nice error messages
            std.process.exit(65);
        },
    };

    if (command == .run) {
        var env = Evaluator.init(alloc);
        defer env.deinit();
        env.run(stmts.items) catch |err| switch (err) {
            error.ValueError,
            error.IoError,
            error.VariableIsNotDeclared,
            => {
                stderr.print("ValueError\n", .{}) catch {};
                std.process.exit(70);
            },
            else => return err,
        };
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
