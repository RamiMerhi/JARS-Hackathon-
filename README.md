# AI-Powered Complaint Triage System

This project is a prototype for an **AI-powered complaint triage system** built for the **Consumer Protection Directorate at Lebanon’s Ministry of Economy and Trade**.

The goal is to help supervisors review citizen complaints more efficiently by automatically classifying complaints, estimating urgency, identifying high-risk establishments, and prioritizing inspection actions.

---

## Problem

Citizens submit complaints about restaurants, supermarkets, shops, butcheries, pharmacies, and other establishments.

Complaints may include:

* Expired food or products
* Hygiene violations
* Food poisoning or unsafe goods
* Price fraud
* Licensing issues
* Product quality problems
* Poor service

In the current process, complaints may enter the same queue even when some are much more urgent than others. Also, the priority selected by citizens may not always reflect the real risk.

For example:

```text
A citizen may mark a serious food poisoning complaint as LOW,
while another citizen may mark a minor service complaint as HIGH.
```

This system helps detect the real urgency of each complaint.

---

## Solution

The system uses a local AI engine to process each complaint and send the result to a supervisor dashboard.

For each complaint, the system can:

* Predict the complaint category
* Estimate the real urgency
* Assign a priority score
* Match the complaint to an establishment
* Use the establishment’s risk zone and violation history
* Detect mismatch between citizen priority and AI priority
* Recommend an action for supervisors

---

## Main Features

### Complaint Classification

The system reads the complaint text and predicts its category.

Example:

```text
Complaint:
"The restaurant served expired chicken and customers became sick."

Predicted category:
Health & Food Safety
```

---

### Priority Detection

The system estimates the true priority of the complaint.

Priority levels:

* CRITICAL
* HIGH
* MEDIUM
* LOW

---

### Establishment Risk Lookup

The system checks the establishment database and uses:

* Establishment name
* Province
* Risk zone
* Previous violations
* Open complaints

Risk zones:

* GREEN: low risk
* YELLOW: medium risk
* RED: high risk

---

### Priority Mismatch Detection

The system compares:

```text
Citizen-selected priority
vs
AI-assigned priority
```

Example:

```text
Citizen priority: LOW
AI priority: CRITICAL
Mismatch: YES
```

This helps supervisors identify complaints that were under-reported or over-reported.

---

### Supervisor Dashboard

The dashboard allows supervisors to:

* View complaints sorted by priority
* Filter complaints by priority, province, category, zone, and status
* See matched establishments and risk zones
* View AI reasoning and recommended actions
* Update complaint status

Supported statuses:

* New
* Assigned to Inspector
* Under Review
* Resolved

---

## AI Approach

This project uses a **local machine learning pipeline**.

It does not use external AI APIs such as:

* OpenAI API
* Claude API
* Gemini API

The current model uses:

```text
TF-IDF + local supervised classifier
```

The AI is trained on the provided complaint data.

The system uses two models:

1. **Category model**
   Predicts what type of complaint it is.

2. **Priority model**
   Predicts the initial urgency using complaint text and establishment information.

After the model prediction, the system applies required safety rules for risk zones.

---

## System Architecture

```text
Complaint Form
      ↓
React Dashboard
      ↓
FastAPI Backend
      ↓
Local AI Model
      ↓
Establishment Matching
      ↓
Priority Rules
      ↓
Triage Result
      ↓
Supervisor Action
```

---

## Tech Stack

### Backend

* Python
* FastAPI
* scikit-learn
* pandas

### Frontend

* React
* TypeScript
* Vite
* Tailwind CSS

### Data

* Complaint CSV file
* Establishment CSV file

---

## Project Structure

```text
project/
├── backend/
│   ├── app/
│   ├── models/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   └── package.json
│
└── data/
    ├── consumer_complaints.csv
    └── establishments.csv
```

---

## Demo Example

Example complaint:

```text
Subject:
Expired chicken at restaurant

Message:
The restaurant served expired chicken and several customers became sick and started vomiting.

Citizen priority:
LOW
```

Expected result:

```text
Predicted category:
Health & Food Safety

AI priority:
HIGH or CRITICAL

Mismatch:
YES
```

This shows that the system does not blindly trust the citizen-selected priority.

---

## What Makes This Useful

The system helps the Ministry:

* Prioritize urgent complaints
* Identify risky establishments
* Detect repeat offenders
* Reduce random inspection inefficiency
* Support supervisors with clear AI reasoning
* Improve consumer protection response

---

## Limitations

This is a prototype and not a full production system.

Future improvements may include:

* More real complaint data
* Arabic and French language support
* Authentication for supervisors
* Permanent database storage
* Inspector assignment by name
* File and image attachment support
* More advanced multilingual AI models

---

## Short Summary

This project is a local AI complaint triage system that helps supervisors classify complaints, estimate urgency, detect priority mismatches, and prioritize inspections more efficiently.
