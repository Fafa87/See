# Foreground extraction 
Here I develop methods for foreground-background separation.

## Static background subtraction

This is the approach assuming that the background is static. 
It is a fairly well understood subject (see the implementation in OpenCV).

Here I will attempt to build something that is very fast, precise and easily tunable.