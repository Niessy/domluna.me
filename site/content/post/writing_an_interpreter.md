+++
date = "2015-07-03T17:33:17-04:00"
title = "Writing an Interpreter. Starting Simple."
tags = ["Go", "interpreters", "compilers", "Brainfuck"]
+++

> "If you don't know how compilers work, then you don't know how computers work. If you're not 100% sure whether you know how compilers work, then you don't know how they work." - Steve Yegge

At first glance compilers appear incredibly intimidating. They do all sorts of black magic that you don't really want to question. All you know is if you type some text and run your compiler, then a nice program that runs on a computer (hopefully) is spit out.

So what's this black magic? Well, actually, I'm not really going to cover that part. That part is an incredibly deep, scary rabbit hole dealing with all the optimizations that take you from an AST to blazingly fast native code. This is what projects like [LLVM](http://llvm.org/) do. Languages like Swift, Rust and Julia, Terra all use LLVM to optimize their code. If you can get your program into an LLVM AST, LLVM will take care of you that rest of the way.

We're going to cover the not so black magic parts.

Lexers, Parsers, ASTs (Abstract Syntax Tree) / IRs (Intermediate Representation) brought out through the implementation of a Brainfuck interpreter. 

These parts are suprisingly approachable. They're also incredibly powerful and in many ways more interesting than the black magic going on with LLVM. 

## The Setup

So why Brainfuck? Since this is for educational purposes I went for the simplest way to get the most educational value. A small subset of Lisp would also be fine, but Brainfuck is even simpler.

The goal isn't to write the shortest or fastest implementation either, but to create a structured way to think about these concepts for future development.

Here's the full [implementation](https://github.com/domluna/brainfuck-go).

With that let's get to it.

## ASTs / IRs

First thing to note here is that as AST is really just an IR. An IR can be though be of representation for your language before it becomes the lowest form (usaully native code). ASTs are the simplest form of IR and referred to the most.

Ok, so in order to form an AST we need some sort of a grammar. The grammar formally describes the structure of the language. In the case of Brainfuck we have 8 actions, all of which are represented by a single character.

So here's our grammar.

```

Program ::= List | Instruction
Instruction ::= '>' | '<' | '+' | '-' | '.' | ','
List ::= '[' Program* ']'
```
`|` means or and `*` means 0 or more.

It reads like this: a Program is either a List or an Instruction. An Instruction is one of
the 6 characters (shown above) and a List is 0 or more Programs.

Here's what the character's mean:

```
">" -> Increases the tape head position by 1
"<" -> Decreases the tape head position by 1
"+" -> Increases the byte value of the byte at the tape head by 1.
"-" -> Decreases the byte value of the byte at the tape head by 1.
"." -> Writes the byte at the tape head to the `output`.
"," -> Reads the byte from the `input` and stores it at tape head.
"[" -> Start of a loop body.
"]" -> End of a loop body.
```

That's it. Here's program that outputs "Hello World!\n".

```
++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.
```

So now unto the implementation. 

program/tape.go:

```go
// Tape represents the tape a Brainfuck program executes on.
type Tape interface {

    // Moves the tape head i spots forward or backward,
    // depending on i.
    MoveHead(i int)

    // Adds i to the byte value at the tape head.
    // This should deal with over/under flows by wrapping.
    AddToByte(i int)

    // Set the byte at the tape head to b.
    SetByte(b byte)

    // Return the byte at the tape head.
    GetByte() byte

    // Get the position of the tape head.
    GetHead() int

    // Set the position of the tape head to i.
    SetHead(i int)
}
```

program/instruction.go

```
// Instructions represents a Brainfuck instruction.
type Instruction interface {
    // Evaluate the instruction on the Tape.
    Eval(t Tape, in io.ByteReader, out io.ByteWriter)

    // String representation of the instruction.
    String() string
}
```

In Go, all a type needs to do be a particular interface is to satisfy it; meaning implement the methods. The interfaces allow us to play around with different execution models on various Tapes. We could have a Tape where the increment/decrement increases only move by 1 or we could make a Tape where we exploit that the fact that we can merge increment/decrement instructions into 1 instruction. For example we can directly move the tape head 10 spaces at once instead of moveing it 1 space 10 times. The key takeaway is that we can interchange the implementations like legos. 

See `program/optimize.go` for the optimization.

Ok! So now that we have our AST down so need to make sense of the characters in our file.

## Lexers

Lexers read in character streams and output tokens. These tokens some sort of a value in our language. In Brainfuck we have 8 characters that have meaning. ">", "<", "+", "-", ".", ",", "[", "]". These are our tokens. All others characters we could care less, they're ignored.

Here's how we define tokens.

lex/lex.go:

```
type Type int

// EOF being the zero value has nice implications
// for when we close our Token channel down the line.
const (
    EOF Type = iota // zero value
    NewLine
    IncTape   // '>' increment tape head
    DecTape   // '<' decrement tape head
    IncByte   // '+' increment byte value at tape head
    DecByte   // '-' decrement byte value at tape head
    WriteByte // '.' write byte to output
    StoreByte // ',' store byte from input to tape header
    LoopEnter // '['
    LoopExit  // ']'
)

// Token represents a tokenized byte.
type Token struct {
    Type    Type
    Line    int
    Pos     int
    ByteVal byte
}
```

If you're not familiar with Go think `Type` as an Enum. They're not the same but for our purposes here they work similar enough. The `Token` type is a wrapper around the byte value and it's `Type`. We send these tokens over to the Parser.

A good way to think of the Lexer is as a state machine. The state we're in depends on the token we're reading and our previous state. In the case of Brainfuck, this is a bit much since each token is comprised of just 1 character. Nonetheless, it's a very nice way to think about it.

For more these resources by Rob Pike are nothing short of superb.

1. [APL Interpreter](https://github.com/robpike/ivy). This is a great starting point for an interpreter implementation.

2. Talk about Lexical scanning:

<iframe width="560" height="315" src="https://www.youtube.com/embed/HxaD_trXwRE" frameborder="0" allowfullscreen></iframe>

## Parsers

The Parser is where we begin to form the structure of our program. Parsers create our initial AST.
Here's the relevant bit for the Brainfuck interpreter.

parse/parse.go:
```go

// nextInst creates the next Instruction based on tok's Type.
func (p *Parser) nextInst(tok lex.Token) program.Instruction {
    switch tok.Type {
    case lex.IncTape:
        return program.InstMoveHead{1}
    case lex.DecTape:
        return program.InstMoveHead{-1}
    case lex.IncByte:
        return program.InstAddToByte{1}
    case lex.DecByte:
        return program.InstAddToByte{-1}
    case lex.WriteByte:
        return program.InstWriteToOutput{}
    case lex.StoreByte:
        return program.InstReadFromInput{}
    case lex.LoopEnter:
        return p.parseLoop()
    case lex.LoopExit:
        return nil
    }
    panic("parse: unreachable")
}

func (p *Parser) parseLoop() program.Instruction {
    insts := make([]program.Instruction, 0)
    for tok := p.next(); tok.Type != lex.EOF; tok = p.next() {
        i := p.nextInst(tok)
        if i == nil { // exit loop
            break
        }
        insts = append(insts, i)
    }
    return program.InstLoop{insts}
}
```

The interesting part here is `parseLoop`, everything else is a one off. A list of instruction is created until the `LoopExit` token is met, this list is the loop body. If we encounter a `LoopEnter` token before the `LoopExit` token for the current loop we recurse.

All that's left is to evaluate the program. Suppose `insts` is our final list of instructions. With`in` and `out` as our input and output streams respectively and `tp` satisfying the Tape interface.

```go
for _, i := range insts {
    i.Eval(tp, in, out)
}
```

## Outro

To recap:

- Lexers reads in character streams and output tokens which have **meaning** in our language.
- Parsers take these tokens as input and form an AST, the **structure** of our program.
- ASTs are one form of an IR. An AST can be manipulated to optimize a program in some way before it's executed or sent further down the pipeline. For example, it's desirable for an AST to be translated to [SSA](https://en.wikipedia.org/wiki/Static_single_assignment_form) form.

An important note before we end. These components are not limited to making new programming languages! Other popular applications include templating languages and [Transpilers](https://en.wikipedia.org/wiki/Source-to-source_compiler).

If you've ever done anything web related I would be shocked if you haven't come across a templating language. This blog itself is generated with the Go text/template package. In a nutshell, they let you programatically generate code; in the case of this blog HTML.

Transpilers are kind of like compilers but instead of targetting a system architecture, they target a source language. They can act as an superset of the target language (provide additional feaures and/or polyfills) or it can just be a syntantic change. 

Examples of the former include SASS (target CSS) and Jade (target HTML). More recent examples target JS. These include Typescript and Babel.js. Typescript adds type information while Babel.js allows to write ES6 (ES2015?) and take care of polyfills and that whole headache.

The canonical example of the latter is probably CoffeScript. CoffeeScript adds no additional features or polyfills (as far as I know). It's a purely syntactic change. It allows you to write Ruby-esque code and converts it to JS. I guess the Ruby folks just didn't like writing JS code. An interesting bit here is that CoffeeScript inspired some of the new syntax in ES6, most notably the => function notation.

If you're further interested transpilers (they are pretty cool!) then I recommend this talk from Jeremy Ashkenas.

<iframe width="560" height="315" src="https://www.youtube.com/embed/DspYurD75Ns" frameborder="0" allowfullscreen></iframe>
