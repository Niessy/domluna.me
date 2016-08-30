+++
date = "2015-08-04T14:43:51-04:00"
title = "Fun with FFI: Rust and Go"
tags = ["Go", "Rust", "FFI", "Monte Carlo"]

+++

> By FFI I'm typically talking about C FFI, that is we can call C through
the FFI through another language, usually a high-level one.

FFI (Foreign Function Interface) is pretty cool stuff. If you think about it
it's how we can justify using higher-level languages. When we need that extra
performance we can dig down and plug into C or Fortran (mostly legacy numerical
stuff).

Without FFI Node.js wouldn't be fast and the Python or R data stack wouldn't be
feasible. Pandas and Numpy, probably the 2 most essential data libraries for
Python are essentially wrappers around C code.

So there's a problem here; we have to write this low-level stuff in C and writing
C code is really hard, especially concurrent/parallel code!

Wouldn't it be nice if we could not use C? Well good news it's 2015 and we
have options!

## Options

We can call upon the help of some new kids, Rust and Go. For Go people, yes
we can now use Go for this kind of stuff, Go 1.5 lets us make shared libraries.
Rust aiming to take C/C++'s lunch money naturally has nice FFI goodness.

I should note we're still using the C FFI, it's just automatically generated
from the Rust and Go tooling. We write Rust and Go code and it interfaces with
C which in turn interfaces with whatever we want.

With that being said, what's our little experiment going to be? Parallel
Monte Carlo Pi estimates of course! [Code + setup instructions](https://github.com/domluna/fun_with_ffi).

![Pi Square](/images/fun_with_ffi/square.png)

So from the above image it's clear the area of the square is (2)^2 = 4. To estimate
Pi we create a lot of random points in the square, if the point is in the circle
we increase the hit count. At the end we do 4.0 * (hit count / total points)

This is example of an algorithm that's embarrassingly parallel. So the parallel
part, just a run a ton of these in parallel and average of the results.

## Results

These are some of the timings I got on my machine (late 2011 MBA).

Python:

```
time python monte.py  -l=python 1000 10000

Running 1000 simulations with 10000 needles in "python".
Estimate Pi = 3.1420776
       18.41 real        18.02 user         0.16 sys
```

Go:

```
time python monte.py  -l=go 1000 10000

Running 1000 simulations with 10000 needles in "go".
Estimate Pi = 3.1413148
        4.82 real        15.15 user         0.49 sys
```

Rust:

```
time python monte.py  -l=rust 1000 10000

Running 1000 simulations with 10000 needles in "rust".
Estimate Pi = 3.1423152
        1.42 real         0.48 user         1.84 sys
```

So I'm not sure why the Go result is slower than the Rust result. Maybe it's
just that Rust is faster than Go, could be. Go also has a GC, not sure how that
plays out here. This is just loops for the most part so I wouldn't expect one
to be much slower than the other.

Go 1.5 is reported to be a bit slower than 1.4 since not everything has been
optimized yet from the great C purge; that might be it. I would compare with 1.4
buttttttt I can't so that's that!

On the bright side I have "4 cores" (yay hyperthreading) on my machine and we
got ~4x speedup at minimum so that's nice!

## Outro

So you might be saying "Hey, there's no C version to compare this to!"

That's true and it's a very fair criticism. My reasoning:

1. I **really** didn't want to write C, it's just not fun.
2. There's already a bunch of evidence out there that shows these languages are
comparable to C performance wise. Go is within a small constant factor and Rust
has that *zero-cost abstraction* thing going for it so it's right there with C/C++.

I think that unless you have a very good reason for using C for FFI things it's
time to give the new kids a spin.
