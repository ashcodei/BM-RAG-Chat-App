# RAG from Scratch – My Little PDF Brain

This project started because I wanted to **actually understand how RAG works**, not just glue together LangChain blocks and pretend I do.

So I built a small system that:

- Takes in PDFs
- Chunks them
- Embeds them
- Builds its **own index (with PCA whitening)**
- Lets me chat with those docs via a custom React frontend

It’s not meant to be a PRODUCT. It’s more like: *“here’s how I understood and implemented a RAG system end-to-end, with real code.”*
