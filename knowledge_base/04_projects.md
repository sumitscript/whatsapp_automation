---
title: Projects Portfolio
tags: projects, project, portfolio, built, build, apps, application, github, repository, side project, hackathon, ai, rag, whatsapp bot, extension, website, sumit
aliases: project kya banaya, kya banaya hai, projects batao, kaunse project, apps banaye, banaya, प्रोजेक्ट, परियोजना, बनाया, neo ai, neoai, faah, faaah, fahh, faahhh, dna matcher, trainer connect, viral whisper, pmgd, train project, railway project, gesture project
---

# Projects Portfolio

I've built AI assistants (NeoAI WhatsApp support genie, ViralWhisper), an AI train traffic control system, enterprise automation at Exxat, a published VS Code extension (Faahhh Terminal Sound), algorithmic and computer-vision projects, web platforms (Trainerconnect, Covid Helpdesk) and my personal portfolio.

## NeoAI — iamneo WhatsApp Support Genie (AI RAG Chatbot, 2025)

I built NeoAI, an AI-driven WhatsApp bot that acts as a support genie for iamneo – An NIIT Venture, during my 2025 impact training.
- It gives instant, accurate answers about iamneo products and services (neocoder.io, NeoHire, neocolab, NeoExam, NeoPAT — formerly Examly), Parul Impact Training, student issues, courses, corporate collaborations and college partnerships.
- Stack: whatsapp-web.js (Node.js) for WhatsApp, a local Mistral LLM via Ollama, RAG with LangChain and ChromaDB, nomic-embed-text embeddings via Ollama, and a Python backend (Flask in the early build, later a FastAPI microservice).
- Its knowledge base came from public information about iamneo and internal Telegram group chats on issue and error resolution.
- It handles out-of-scope questions politely ("I apologize, but based on the information I have about iamneo.ai, I cannot answer that question directly.") so answers stay grounded.

## NeoAI — Architecture and Results

- Microservices architecture linking a FastAPI (Python) RAG service with a Node.js whatsapp-web.js gateway, built for high concurrent traffic.
- The RAG pipeline gives context-grounded answers from private enterprise knowledge bases.
- Results: about 90% lower response latency and 10x better scalability, with strong business interest.
- I jokingly credit my friend Mehul Gosavi, whose phone "survived more test messages than he'd care to admit."

## AI-Powered Precise Train Traffic Control System (Konkan Railway)

- Problem: railway traffic management on the Konkan Railway network, a large-scale combinatorial optimization problem.
- I engineered and backtested an AI-powered traffic control system for dispatching and routing.
- Stack: React + TypeScript frontend for real-time monitoring, and a Flask (Python) backend with the machine learning optimization model.
- It gives train controllers automated decision support: real-time optimization of train precedence, dynamic route planning and conflict resolution.
- KPIs: maximizing section throughput and improving on-time performance.

## Faahhh Terminal Sound — My VS Code Extension (SumitScript)

People kept joking online about a "faahhh" terminal extension for VS Code, so I built one.
- It plays a sound when something goes wrong in the terminal (and can also play on successful completion). It covers terminal executions, build tasks and debugger states.
- Features: bring your own .wav or .mp3, adjustable volume (0–100), configurable delay, customizable keywords (error, warning, crash, exception, failed, panic), debounced playback so a wall of errors plays once, works on Windows, macOS and Linux, zero npm dependencies.
- Marketplace ID: SumitScript.faahhh-terminal-sound
- VS Code Marketplace: https://marketplace.visualstudio.com/items?itemName=SumitScript.faahhh-terminal-sound
- VsixHub: https://www.vsixhub.com/publisher/38999/
- Fun story: while packaging the .vsix I started designing the logo, an error popped up, and my laptop blasted a loud "faaaahhhh" — proof it worked.

## Exxat Internal Productivity and Automation Platform

- Stack: Next.js, TypeScript, ExcelJS, MongoDB
- I single-handedly architected, built, deployed and manage this enterprise tool, adopted team-wide at Exxat.
- It processes 85,000+ records across 9 datasets; 23 recurring reports went from 6–8 hours to under 5 minutes.
- It includes an on-device support chat assistant with 3,161+ interactions.
- I presented it to Exxat C-suite executives about 2 months after joining.

## DNA Sequence Matcher — Bioinformatics with Next.js, MongoDB, KMP and Levenshtein

- A challenge from my teacher N Narasimha Pillai Sir during Impact Training: compare an input DNA sequence with stored sequences from different species and find exact matches and percentage similarity.
- DNA sequences average about 156,000 characters, so brute force is impractical.
- Exact matching uses the Knuth-Morris-Pratt (KMP) algorithm with an LPS (Longest Prefix Suffix) array. Similarity matching uses Levenshtein Distance (insertions, deletions, substitutions) to find the closest species.
- Stack: Next.js frontend with pre-built UI components, MongoDB storage, tested on dummy sequences. It was my first proper Next.js project.

## Hand Gesture-Controlled Image Viewer — Python, OpenCV, MediaPipe

- Inspired by a LinkedIn post, I learned Python computer vision from scratch (Python was not my primary stack) and built a touchless image viewer.
- Features: pinch-to-zoom scaling, swipe gestures to switch images and virtual cursor tracking by finger, with no physical peripherals.
- Stack: Python, OpenCV, MediaPipe, real-time image processing.
- Guided by N Narasimha Pillai Sir and Suraj Mourya Sir. It's one of my most-viewed LinkedIn posts (about 4,783 impressions).

## Trainerconnect — Trainer Marketplace (HTML, CSS, JavaScript, Flask, MongoDB)

- An online portal to hire trainers or become one: fitness, language and skill-based trainers.
- Features: location, skill and availability filters, trainer reviews and customer testimonials.
- Planned: in-app chat, better filters and a stronger matchmaking algorithm.
- Stack: HTML5, CSS3, JavaScript, Flask (Python), MongoDB. We built it as a team project under the guidance of N Narasimha Pillai Sir. Its domain is no longer active.

## ViralWhisper — Local AI Social Engagement Tool

- Repository: https://github.com/sumitscript/ViralWhisper
- An AI-powered social engagement tool for solo founders. It finds relevant posts through keyword monitoring, writes meaningful contextual replies and subtly promotes your project.
- Runs fully locally using Ollama LLMs.

## Covid Helpdesk Website

- A real-time community resource and helpdesk portal I built during the Covid pandemic.
- It brought together emergency contact information, hospital availability and verified healthcare resources.
- It was featured in the newspaper Hindustan News.

## PMGD — Predictive Modelling for GNIDA's Developmental Strategies (Sharda University Hackathon)

- Built at the 6th Hackathon Sharda University by my Team NPM_BUG (Sumit Soni, Aditya Patil, Jay Likhar, Suryansh Raghuwanshi). We were finalists.
- An AI-powered smart-city platform for the Greater Noida Industrial Development Authority (GNIDA).
- Sustainable land allocation, data-driven infrastructure planning insights, environmental impact assessment and AI-driven urban growth forecasting.

## Samsung Solve for Tomorrow Idea — Mental Wellness and Motivation Platform

- My concept for Samsung Solve for Tomorrow: a mental wellness, motivation and productivity tool for students, university youth and young adults.
- I ran a short survey (5–8 minutes, no personal info) to learn what users need, and promised early supporters free premium access at launch.
- It reached the Top 100 nationwide and Top 25 in its domain (see Achievements).

## My Personal Portfolio Website — sumitscript.com

- Live site: https://www.sumitscript.com
- Repository: https://github.com/sumitscript/Portfolio
- A clean developer portfolio of my technical projects, skills, articles and interactive demos.
- Other repositories on my GitHub include Sumit_nimbus.

## Early Web Projects — 30 Days #CodeHostage Challenge

During my 30-day MERN learning challenge, I built small web projects:
- A personal resume page in HTML and CSS, with an FAQ section using the details element (Day 5)
- A job application form with many HTML input types, fieldsets and legends, later restyled with CSS (Day 4)
- A Recipe Website using 2D transforms and CSS animations (Day 20)
