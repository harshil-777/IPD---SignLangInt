"""Core domain package for the sign language interpreter.

Kept dependency-light: submodules that need heavy libraries (TensorFlow,
MediaPipe) only import them inside the modules that actually use them, so
importing a lightweight module (e.g. ``sign_language.fir``) never pulls in
the rest of the stack.
"""
