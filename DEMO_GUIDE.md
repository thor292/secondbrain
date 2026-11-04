# SecondBrain Demo and Testing Guide

How to run the demo, what data is loaded, how to test each feature, and how
to explain it to a client.

---

## 1. Before you start

1. **Turn on the VPN.** The AI models (LiteLLM) are only reachable through it.
2. From the project folder, start the databases and the server:

   ```bash
   cd ~/Desktop/secondbrain
   docker compose up -d --wait
   uv run uvicorn secondbrain.api:app --port 8765
   ```

3. Open **http://localhost:8765** (chat) and **http://localhost:8765/graph-view** (graph).
4. The admin token for the whole-SecondBrain view is `SB_ADMIN_TOKEN` in `.env`
   (`dev-admin-token` locally).

**To reset everything** to the original demo data (takes about a minute):

```bash
uv run python -m secondbrain.schema --reset
./scripts/load_examples.sh
```

Reset before a client demo, so notes added while practising don't show up.

---

## 2. What data is loaded

### Companies

| Company | What it is | Shares with the root |
|---|---|---|
| **Lumen Home** | Online home decor and furniture brand, Portland, about 120 staff | Yes, anonymized patterns only |
| **Brightpath Logistics** | Small logistics company (warehouses in Dallas and Atlanta) | No |

### Lumen Home people and what they can see

| Person | Role | Department | Can see |
|---|---|---|---|
| **Maya Chen** | CEO | (company) | Everything in Lumen, including restricted |
| **Rachel Kim** | CFO | Finance | Everything in Lumen, including restricted |
| **Arjun Mehta** | Head of Support | Support | Support (incl. restricted) + company |
| **Nina Alvarez** | Support agent | Support | Support + company **+ Operations (internal)** |
| **Leo Park** | Marketing lead | Marketing | Marketing (incl. restricted) + company **+ Merchandising (internal)** |
| **Sofia Rossi** | Content marketer | Marketing | Marketing + company |
| **Daniel Okafor** | Operations manager | Operations | Operations (incl. restricted) + company |
| **Ivy Thompson** | Warehouse coordinator | Operations | Operations + company |
| **Tom Bauer** | Finance analyst | Finance | Finance (not restricted) + company |
| **Hana Sato** | People partner | People & HR | People & HR (incl. restricted) + company |
| **Omar Haddad** | Merchandiser | Merchandising | Merchandising + company |
| **Karim Nasser** | Operations director | Brightpath | Only Brightpath |

"Lead" roles see their department's restricted data; "member" roles don't.
Executives (CEO, CFO) see the whole company. Nina and Leo have one **extra
grant** each into another department, the "through the parent, with
permission" rule.

### Lumen Home documents (18)

| Brain | Document | Format | Sensitivity |
|---|---|---|---|
| Company | Employee Handbook (leave, remote work, expenses with hotel cap table, code of conduct) | **Word** | Internal |
| Company | Returns and Refunds Policy | **PDF** | Internal |
| Company | Company Overview (mission, FY2026 goals, tools) | Markdown | Internal |
| Company | Shipping Policy (free shipping over $75, delivery times) | Markdown | Internal |
| Support | Customer Support Playbook (response times, VIPs, goodwill credits, escalation) | Markdown | Internal |
| Marketing | Holiday Campaign Plan 2026 (key dates, offers, channels) | **Word** | Internal |
| Marketing | Brand Voice Guide | Markdown | Internal |
| Marketing | Marketing Budget and CAC | Markdown | **Restricted** |
| Operations | Fulfillment SOP (warehouse, cutoffs, carriers, peak season) | **PDF** | Internal |
| Operations | Carrier Contracts Summary | Markdown | **Restricted** |
| Finance | Spending and Approval Policy | Markdown | Internal |
| Finance | Q3 FY2026 Financial Results | **PDF** | **Restricted** |
| People & HR | Hiring Process | Markdown | Internal |
| People & HR | Onboarding Checklist | Markdown | Internal |
| People & HR | Compensation Bands 2026 | Markdown | **Restricted** |
| Merchandising | Pricing and Discount Policy | Markdown | Internal |
| Merchandising | Assortment Notes Q4 (bestsellers, stock risks, suppliers) | Markdown | Internal |

Brightpath has a handbook (20 days leave) and operations notes.

### Notes from Lumen's people (20)

Examples, saved through the normal remember → long-term flow:

- **Daniel (Ops):** UPS delays in the Midwest; **free shipping threshold drops from $75 to $60 from November 1**; second warehouse shift.
- **Nina (Support):** VIP customer Priya Shah's second damaged lamp, $25 credit; Priya prefers SMS.
- **Arjun (Support):** WISMO tickets up 35% after the UPS delays; approved a $120 credit.
- **Leo (Marketing):** creator partnership; **loyalty early access moved from Nov 24 to Nov 23**.
- **Omar (Merchandising):** Aria sofa back in stock on December 2.
- **Rachel (CFO, restricted):** board wants Black Friday discounts kept at 20%.
- **Hana (People):** two open roles; Arjun is the hiring manager.

### What maintenance already did

Four document sections were marked **partly updated** by newer notes: the
Shipping Policy (threshold), Holiday Campaign key dates (early access),
Assortment stock risks (sofa back in stock), and one of Priya's notes.

### What the root learned

**8 anonymized lessons** from Lumen, for example *"Lower the free-shipping
threshold during peak season to increase conversion, then restore it
afterwards to protect margins."* No names, numbers or company details.

In total: **97 memories** across 10 brains.

---

## 3. How to explain SecondBrain (2 minutes)

> **SecondBrain is a company's memory you can talk to.**
>
> Every company gets its own brain, and every department can have its own
> brain inside it. They're connected like a tree, with our organization's
> brain at the root.
>
> Each brain remembers documents, facts, events and information about people.
> When someone asks a question, their department's brain answers first. If it
> doesn't know, the company brain answers. It never looks into another company.
>
> Everyone only sees what they're allowed to see, and that is enforced by the
> database itself, not just the application.
>
> The brain also keeps itself up to date: when a newer note changes a fact,
> it uses the new fact. And companies that agree can share **anonymous
> lessons** with the root, so every company benefits without anyone's data
> leaving their company.

---

## 4. Feature by feature: test it, then explain it

Each test says **who to pick**, **what to do**, **what you should see**, and
**what to say**.

### F1. Answers from your department

- **Pick:** Nina (Support agent)
- **Ask:** *Can I give a customer a 50 dollar credit without asking anyone?*
- **Expect:** Green badge **"Answered from your department"**: no, agents can give up to $25; a team lead can approve up to $100. Click **sources** to see the Support Playbook.
- **Say:** "Nina's question is answered from her own department's knowledge, with the source it came from."

### F2. Falling back to the company brain

- **Pick:** Nina
- **Ask:** *How many vacation days do I get?*
- **Expect:** Orange badge **"Answered from your company"**: 25 days, up to 5 carried over until March 31.
- **Say:** "Support doesn't know about leave, so the company brain answered. It goes up one level, never sideways into other companies."

### F3. Conversation memory

- **Pick:** Nina, right after F2
- **Ask:** *and can I carry any over?*
- **Expect:** An answer about the 5 carry-over days, with **"Understood as: How many vacation days can I carry over…"** under it.
- **Say:** "It remembers the conversation, so follow-up questions work like they would with a colleague."
- Use **New conversation** to start fresh.

### F4. Permissions: restricted data

- **Pick:** Tom (Finance analyst). **Ask:** *What was gross margin in Q3?*
- **Expect:** **"Not found in your brains."**
- **Pick:** Rachel (CFO). **Ask** the same.
- **Expect:** **46 percent**, down from 48%, from the Q3 Results PDF.
- **Say:** "Same department, same question, different answers. Tom isn't cleared for restricted finance data; the CFO is."

### F5. Permissions: access between departments

- **Pick:** Nina. **Ask:** *What's the free shipping threshold right now?*
- **Expect:** **$60**, reduced from $75 on November 1.
- **Pick:** Sofia (Marketing). **Ask** the same.
- **Expect:** **$75**.
- **Say:** "The $60 change is a note in Operations. Nina has permission to read Operations, so she gets the update. Sofia doesn't, so she still gets the official policy. Nobody sees data they aren't allowed to, and nobody loses information because of a note they can't read."

### F6. Isolation between companies

- **Pick:** Karim (Brightpath). **Ask:** *How many days of annual leave do we get?*
- **Expect:** **20 days**, never Lumen's 25.
- **Say:** "Each company's knowledge is completely separate."

### F7. PDF and Word documents

- **Pick:** Sofia. **Ask:** *What is the hotel cap for a manager?*
- **Expect:** **$190 per night**. This comes from a **table inside the Word handbook**.
- **Pick:** Nina. **Ask:** *Do customers need to send back a damaged item?*
- **Expect:** Not for items under $150; a photo is enough. This comes from the **Returns Policy PDF**.
- **Say:** "It reads the documents companies actually have: PDF and Word, including tables."

### F8. Remembering new information

- **Pick:** Arjun (Head of Support), open the **Remember** tab.
- Type each of these and click **Remember**:
  - *Met the Bluebird Stores buyer today, they want delivery before December 10* → shows **episodic** (an event)
  - *Our standard payment terms for wholesale orders are net 45* → shows **semantic** (a fact)
  - *The Bluebird buyer prefers WhatsApp over email* → shows **user** (about a person)
- Click **Save to long-term**.
- Back on **Ask**: *When does Bluebird need delivery?* → **before December 10**.
- **Pick:** Leo (Marketing). Ask the same → **Not found**: Arjun's notes stay in Support.
- **Say:** "New knowledge goes into short-term memory, gets sorted automatically, and is saved to long-term memory, only visible to the right people."

### F9. Memory keeps itself up to date (live)

- **Pick:** Arjun. **Remember:** *From today support agents can give goodwill credits up to 40 dollars without approval*. Click **Save to long-term**.
- Wait about **10 seconds** (maintenance runs in the background).
- **Pick:** Nina. **Ask:** *How much credit can I give without approval?*
- **Expect:** **$40**, mentioning it was raised from $25.
- Already loaded: **Leo**, *When does loyalty early access for Black Friday start?* → **November 23**, moved from the plan's November 24.
- **Say:** "When something changes, the brain notices that the new note updates an older fact and uses the newer one. Nothing is deleted; you can still see the history in the graph."

### F10. Graph: what each person can see

- Open **http://localhost:8765/graph-view**, pick **Rachel (CFO)**.
- **Expect:** All six department zones filled, restricted memories with a **red outline**.
- Pick **Sofia**. **Expect:** Most zones empty; she only sees Marketing and company knowledge.
- Click a node (e.g. a department or a person) to see its details and relationships.
- **Say:** "This is the same permission model, made visible. The graph is built by the database for this person, so it can't show more than they're allowed to see."

### F11. Graph: the whole SecondBrain

- Pick **★ Whole SecondBrain (admin)**, enter the admin token, click **Open**.
- **Expect:** The gold **SecondBrain Root** in the middle with its **8 learned lessons** around it; Lumen Home and Brightpath as clusters.
- Faded nodes with an orange or yellow dashed ring are **updated or partly updated** memories; click one to see the **"Partly updated by"** link and the reason.
- **Say:** "This is our view of the whole network. The root holds only anonymous lessons learned from companies that agreed to share."

### F12. Learning at the root

- In the whole view, click any small node right around the root to read a lesson.
- To run learning again (it skips lessons it already has):

  ```bash
  curl -s -X POST http://localhost:8765/admin/learn -H "X-Admin-Token: $(grep '^SB_ADMIN_TOKEN=' .env | cut -d= -f2)"
  ```

- **Say:** "Lessons are written without any names or numbers, then checked by code; anything identifying is rejected. Restricted data is never used, and Brightpath, which doesn't share, contributes nothing. Company users never get answers from the root."

### F13. Onboarding a new company live

1. Make a folder `~/Desktop/nova_demo` with a file `company.json`:

   ```json
   {
     "id": "nova",
     "name": "Nova Tech",
     "sharing": "none",
     "departments": [{"id": "eng", "name": "Engineering"}],
     "users": [{"id": "priya", "name": "Priya Raman (Engineer)", "department": "eng", "role": "member"}]
   }
   ```

   and a file `oncall.md`:

   ```
   # On-call Guide

   ## Rotation
   The on-call rotation changes every Monday at 10am. P1 incidents are escalated to the engineering manager within 15 minutes.
   ```

2. Load them:

   ```bash
   uv run python -m secondbrain.admin onboard ~/Desktop/nova_demo/company.json
   uv run python -m secondbrain.admin import root.nova.eng ~/Desktop/nova_demo/oncall.md --sensitivity internal
   ```

3. **Refresh** the chat. Pick **Priya**, ask *When does on-call change?* → **Monday at 10am**.
4. **Say:** "A new company is live in minutes: one file for its structure, and its documents."

### F14. Proof that the database enforces permissions

For technical audiences. This connects the way the app does and asks for all memories:

```bash
docker compose exec postgres psql "postgresql://secondbrain_app:dev_app_password@localhost/secondbrain" -c "SELECT count(*) FROM memories;"
```

Result: **0**, because no person is set.

```bash
docker compose exec postgres psql "postgresql://secondbrain_app:dev_app_password@localhost/secondbrain" -c "BEGIN; SELECT set_config('app.user_id','sofia',true); SELECT DISTINCT brain_path FROM memories; COMMIT;"
```

Result: only `root.lumen` and `root.lumen.marketing`.

- **Say:** "Even if someone bypassed our application, the database itself would only return what that person may see."

### F15. Automated tests

```bash
uv run pytest
```

Expect **34 passed**. They run on a separate test database.

---

## 5. Questions clients usually ask

| Question | Answer |
|---|---|
| Where is our data stored? | In our PostgreSQL database, separated per company, with access enforced by the database. It can also run on your own servers. |
| Who can see what? | You define departments and roles; each person sees their department, the company level, and anything they're explicitly granted. Show F4, F5, F10. |
| Does our data train the AI? | No. Your documents are stored as memory; the AI reads them to answer, but no model is trained on them. |
| Does our data go to other companies? | Never. Only if you opt in, anonymous lessons with no names or numbers go to our root, and other companies never query it directly. |
| What happens when information changes? | Newer notes update older facts automatically (F9). Nothing is deleted, so there's a full history. |
| Which documents does it read? | PDF, Word, Markdown and text today. Google Drive, Slack and email connectors are planned. |
| How do people log in? | Today the demo uses a user picker. In production it uses your existing sign-in (Google, Microsoft, Okta). |

---

## 6. Know before the demo

- **The VPN must stay on.** Without it, questions fail.
- **Answers take 2 to 5 seconds**, longer for follow-ups and the first question after starting.
- **Reset the data** before a client demo so practice notes are gone.
- The **user picker is a demo stand-in** for real sign-in; mention SSO is planned.
- The **whole-SecondBrain view** shows every company's data. With a real client, only show it with sample companies, never another client's data.
- **Maintenance isn't perfect.** It can mark a section as "partly updated" when it only added detail. That's harmless: nothing gets hidden in that case.
- Only **load real client documents** if the client agrees to them being processed by the AI models behind our LiteLLM proxy.
