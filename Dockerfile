# Use a lightweight Python base image
FROM python:3.9-slim

# Create a non-privileged user for safety
# (Prevents the bot from having 'root' access inside the container)
RUN useradd -m botuser

# Set the working directory
WORKDIR /home/botuser/app

# Switch to the non-privileged user
USER botuser

# This container will wait for a command, or we can run a script directly
# We will pass the code to run via the bot later
CMD ["run", "app.py"]
