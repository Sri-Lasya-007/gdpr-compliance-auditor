🛡️ GDPR Compliance Auditor (OpenEnv Agent)
An LLM-powered autonomous agent designed to evaluate complex data privacy requests, enforce GDPR/CCPA compliance rules, and surgically redact Personally Identifiable Information (PII). Built and optimized for the OpenEnv benchmark platform and ready for deployment on Hugging Face Spaces.

📖 Overview
The GDPR Compliance Auditor tests an AI agent's ability to act as a strict data privacy officer. It receives JSON observations detailing data transfer requests and must output a strict JSON decision to approve, block, or redact specific strings.

The environment features a custom fractional reward system that penalizes over-redaction and rewards precise string-matching redactions based on varying regional privacy laws.

✨ Key Features
6 Privacy Scenarios: Ranges from simple policy violations to complex mixed-compliance rules (e.g., handling medical data vs. fraud scores)
Fractional Reward Engine: Calculates precision and recall on exact-string redactions
Strict Logging Format: Adheres to OpenEnv's strict [START], [STEP], [END] stdout parsing rules for automated grading.
Multi-Mode Ready: Includes both a standalone inference script and a decoupled FastAPI server for independent environment testing.
Hugging Face Optimized: Configured to run cleanly on Docker Spaces with non-root user permissions.

📂 Project Structure
.
├── Dockerfile             # Container definition for Hugging Face Spaces
├── inference.py           # The LLM Agent that processes the 6 privacy tasks
├── models.py              # Pydantic schemas for observations and actions
├── openenv.yaml           # OpenEnv manifest defining tasks and LLM graders
├── pyproject.toml         # Project metadata and dependencies
└── server/                # Environment server directory
    ├── app.py             # FastAPI server and OpenEnv multi-mode entry point
    └── env.py             # Core environment classes and ground truth data


🚀 Quick Start (Local Execution)
To run this project locally, you will need two terminal windows: one to host the environment server, and one to run the AI agent.

1. Install Dependencies
Ensure you have Python 3.10+ installed. Create a virtual environment and install the required packages:
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt  # Or install directly: pip install fastapi uvicorn openai pyyaml pydantic openenv-core

2. Set Environment Variables
The agent requires access to an LLM provider (defaults to Hugging Face's router and Qwen2.5-72B-Instruct).
export HF_TOKEN="your_huggingface_or_openai_api_key"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"

3. Start the Environment Server (Terminal 1)
Boot up the FastAPI server from the server directory to act as the environment gateway.
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

4. Run the Agent (Terminal 2)
In a new terminal window (with your environment variables set), execute the agent to process the OpenEnv tasks. The script will output the strict OpenEnv execution logs to stdout.

# Run a specific task
python inference.py --task task_3

# Or let OpenEnv handle the task injection natively via ENV variables
python inference.py
🐳 Docker Deployment
This project includes a production-ready Dockerfile optimized for Hugging Face Spaces (Port 7860, non-root user).

Build the image:

docker build -t gdpr-auditor .
Run the container:

docker run -p 7860:7860 -e HF_TOKEN="your_api_key" gdpr-auditor
The FastAPI health check will be available at http://localhost:7860/.

📋 Task Breakdown
The agent is evaluated on 6 distinct tasks defined in openenv.yaml:
Task 1 (Easy): Block unauthorized marketing access to financial data.
Task 2 (Medium): Surgically redact PII (Names, Phones) for internal data science logs.
Task 3 (Hard): Navigate mixed-compliance rules (GDPR Deletion vs. Fraud Prevention vs. Medical Data).
Task 4 (Easy): Approve clean, anonymized application telemetry.
Task 5 (Medium): Enforce strict EU GDPR biometric regional transfer restrictions.
Task 6 (Hard): Redact internal employee email leaks from system metadata without blocking valid data.

🛠️ Built With
FastAPI - Environment server framework
Pydantic - Data validation and JSON schemas
OpenAI Python SDK - LLM Routing
OpenEnv - Evaluation framework
