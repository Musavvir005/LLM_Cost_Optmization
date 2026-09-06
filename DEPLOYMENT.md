# 🚀 Full-Stack Deployment Guide (Vercel Frontend + Cloud Backend)

This repository contains:
1. **Frontend**: React + Vite SPA located in `/frontend` (deployed on Vercel at `https://adybot.vercel.app`).
2. **Backend**: Python FastAPI service located in repository root (`api.py`) with ML routing, PyTorch/Transformers, and semantic caching.

---

## 1. Why Vercel Returned `Unexpected token 'T' / 404`

Vercel is hosting the static React frontend. By default, API calls (`/api/route`, `/api/health`) were hitting Vercel's static servers rather than your Python backend, causing Vercel to return its standard HTML 404 page (`"The page cannot be found..."`).

---

## 2. Step-by-Step: Deploy the Backend (Free on Render)

### Step 1: Create a Web Service on Render
1. Sign up / Log in to [Render.com](https://render.com).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository: `Musavvir005/LLM_Cost_Optmization`.

### Step 2: Configure the Web Service
- **Name**: `llm-cost-backend` (or any preferred name)
- **Language**: `Python 3` (or `Docker`)
- **Region**: Closest to you (e.g., Singapore / Frankfurt / Oregon)
- **Branch**: `main`
- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  uvicorn api:app --host 0.0.0.0 --port $PORT
  ```

### Step 3: Set Environment Variables on Render
In the **Environment** tab on Render, add:
- `GROQ_API_KEY` = *(Your Groq API key)*
- `GEMINI_API_KEY` = *(Your Google AI Studio Gemini API key)*
- `ADBON_API_KEY` = `adbon-sec-key-2026-demo`
- `ADBON_SIGNAL_SECRET` = `adbon-signal-sec-2026-demo`

Click **Deploy Web Service**. Once deployed, Render will provide a public HTTPS URL (e.g. `https://llm-cost-backend.onrender.com`).

---

## 3. Connect Vercel to Your Cloud Backend

1. Open your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click on your project (`adybot`).
3. Go to **Settings** → **Environment Variables**.
4. Add a new variable:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://llm-cost-backend.onrender.com` *(use your actual backend URL from Render)*
5. Go to **Deployments** tab → Click the three dots (`...`) on the latest deployment → Select **Redeploy**.

Once redeployed, your Vercel frontend will connect directly to your cloud Python backend, the status indicator will show `🟢 API Online`, and routing queries will execute live!

---

## 4. Alternative: Instant Testing via Localhost (`ngrok`)

If you want to test right now with your local PC running the backend:

1. In your terminal, run the backend:
   ```bash
   python -m uvicorn api:app --host 0.0.0.0 --port 8000
   ```
2. In another terminal, expose port 8000 with ngrok:
   ```bash
   npx ngrok http 8000
   ```
3. Copy the forwarding URL (`https://xxxx-xx.ngrok-free.app`).
4. In Vercel Project Settings → Environment Variables, set `VITE_API_BASE_URL` to that URL and redeploy.
