This synthetic target is only for environment tests; it is not an accepted
production DNS result. It uses 128 uniform cells on [0, 2*pi], a sine mean
profile sampled at cell centers, and a single-mode spectrum. The metadata
uses the accepted-target schema and SHA-256 hashes of the exact CSV bytes
to exercise the production loader without depending on ignored `runs/` files.
