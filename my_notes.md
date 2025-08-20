# Things I learn and/or need to keep in mind

### 🔠 Why do we need chunking?

LLMs have their limited context window (e.g: gpt-4.0-nano has a context window of 1M token), if our document is long (probably about 10000 pages) and we do not break it down, the whole document is going to be vectorized, leading to a vector not being specific about anything (if it's too small, it loses context). Breaking the doc down makes it easier to manage and compare and faster to retrieve. 



