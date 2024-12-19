FROM python:3.9  # Changed to 3.9 for better compatibility

WORKDIR /app  # Changed from / to /app for better practice
COPY . .

# Install numpy first to fix the NaN error
RUN pip install numpy==1.23.5
RUN pip install -r requirements.txt

# Add environment variable for Render
ENV PORT=8080

# Update the CMD to use gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT robo_dashboard:server