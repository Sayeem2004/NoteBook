# Techniques

- **Meet In The Middle** Instead of doing a `O(2^N)` operation, attempt to do two `O(2^(N/2))` operations, and combine the two using faster operation than `O(2^N)`. Usually splitting more than once or twice is not beneficial for this technique.
