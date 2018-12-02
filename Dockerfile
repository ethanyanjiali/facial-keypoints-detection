FROM alpine:latest

RUN sudo apt-get update
RUN sudo apt install git-all
RUN sudo apt-get install build-essential
RUN sudo apt-get install unzip
RUN sudo apt-get install python3-venv