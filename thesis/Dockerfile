# Minimal TeX Live image for compiling the SOICT đồ án tốt nghiệp template.
# Image tag: thesis-tex:latest
#
# Build (run once, from the thesis/ directory):
#   docker build --platform linux/amd64 -t thesis-tex:latest .
#
# Why not the official texlive/texlive image?
#   It works, but is ~5 GB. This image installs only what the SOICT template
#   needs from the texlive-* collections in Debian and is ~1.5 GB.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/texlive/bin/x86_64-linux:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        latexmk \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-plain-generic \
        texlive-fonts-recommended \
        texlive-fonts-extra \
        texlive-science \
        texlive-pictures \
        texlive-bibtex-extra \
        texlive-publishers \
        texlive-lang-other \
        texlive-lang-european \
        texlive-luatex \
        texlive-xetex \
        lmodern \
        cm-super \
        biber \
        ghostscript \
        make \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
