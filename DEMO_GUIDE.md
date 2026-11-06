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

   This also starts **FalkorDB** (the relationship graph, F15).

3. Open **http://localhost:8765** (chat) and **http://localhost:8765/graph-view** (graph).
4. The admin token for the whole-SecondBrain view is `SB_ADMIN_TOKEN` in `.env`
   (`dev-admin-token` locally).

**To reset everything** to the original demo data (takes about two minutes):

```bash
uv run python -m secondbrain.schema --reset
./scripts/load_examples.sh
```

Reset before a client demo, so notes added while practising don't show up.

---

## 2. What data is loaded

The demo world is Indian commerce, in rupees (spec: `docs/INDIA_WORLD.md`).
The demo's today is **Friday 6 November 2026, Dhanteras**; **Diwali is Sunday
8 November**, two days away.

### Companies

| Company | What it is | Shares with the root |
|---|---|---|
| **Desi Living** | Bengaluru D2C home decor brand (Moradabad brass, Jaipur block print, Panipat dhurries, sheesham wood, Khurja ceramics), about 140 staff, ~₹45 crore a year. Sells on its own Shopify store (desiliving.in), Reliance JioMart and 40 DMart shop-in-shops. Warehouses in Hoskote (Bengaluru) and Bhiwandi (Mumbai) | Yes, anonymized patterns only |
| **BharatFleet Logistics** | Pune large-item logistics partner that delivers Desi Living's furniture | No |

### Desi Living people and what they can see

| Person | Role | Department | Can see |
|---|---|---|---|
| **Meera Iyer** | CEO | (company) | Everything in Desi Living, including restricted |
| **Anand Krishnan** | CFO | Finance | Everything in Desi Living, including restricted |
| **Arjun Mehta** | Head of Support | Support | Support (incl. restricted) + company |
| **Neha Kulkarni** | Support agent | Support | Support + company **+ Operations (internal)** |
| **Rohan Kapoor** | Marketing lead | Marketing | Marketing (incl. restricted) + company **+ Merchandising (internal)** |
| **Sana Sheikh** | Content marketer | Marketing | Marketing + company |
| **Vikram Rao** | Operations manager | Operations | Operations (incl. restricted) + company |
| **Priya Nair** | Warehouse coordinator | Operations | Operations + company |
| **Tanvi Shah** | Finance analyst | Finance | Finance (not restricted) + company |
| **Farah Khan** | People partner | People & HR | People & HR (incl. restricted) + company |
| **Kabir Singh** | Merchandiser | Merchandising | Merchandising + company |
| **Karan Malhotra** | Operations director | BharatFleet | Only BharatFleet |

"Lead" roles see their department's restricted data; "member" roles don't.
Executives (CEO, CFO) see the whole company. Neha and Rohan have one **extra
grant** each into another department, the "through the parent, with
permission" rule.

### Desi Living documents

| Brain | Document | Format | Sensitivity |
|---|---|---|---|
| Company | Employee Handbook (earned leave, holidays, remote work, expenses with hotel cap table, code of conduct) | **Word** | Internal |
| Company | Returns and Refunds Policy | **PDF** | Internal |
| Company | Company Overview (channels, products, suppliers, FY2026-27 goals, tools) | Markdown | Internal |
| Company | Shipping Policy (free shipping over ₹999, pincode zones, COD, Diwali shipping cutoff) | Markdown | Internal |
| Support | Customer Support Playbook (WhatsApp first, VIPs, goodwill credits, escalation, damaged brass lamps) | Markdown | Internal |
| Marketing | Diwali Campaign Plan 2026 (key dates, offers, channels) | **Word** | Internal |
| Marketing | Brand Voice Guide | Markdown | Internal |
| Marketing | Marketing Budget and Performance | Markdown | **Restricted** |
| Operations | Fulfillment SOP (warehouses, cutoffs, carriers, peak season) | **PDF** | Internal |
| Operations | Carrier Contracts (BharatFleet rates, what a sofa costs to deliver) | Markdown | **Restricted** |
| Finance | Spending and Approval Policy | Markdown | Internal |
| Finance | Q2 FY27 Financial Results | **PDF** | **Restricted** |
| People & HR | Hiring Process | Markdown | Internal |
| People & HR | Onboarding Checklist | Markdown | Internal |
| People & HR | Compensation Bands FY2026-27 | Markdown | **Restricted** |
| Merchandising | Pricing and Discount Policy | Markdown | Internal |
| Merchandising | Assortment Notes Q3 FY27 (bestsellers, stock risks, suppliers) | Markdown | Internal |

BharatFleet has a handbook (15 days earned leave) and operations notes.

### Notes from Desi Living's people

Examples, saved through the normal remember → long-term flow:

- **Vikram (Ops):** Delhivery festival delays to Delhi NCR and Kolkata; **free shipping threshold drops from ₹999 to ₹699 from Dhanteras (6 November) to 15 November**; second shift and Sunday working until 15 November.
- **Priya (Ops):** 36 damaged Moradabad brass floor lamps from Noor Metalcraft, quarantined in bay Q.
- **Neha (Support):** VIP customer Ananya Reddy's second damaged brass lamp, ₹500 credit; Ananya prefers WhatsApp.
- **Arjun (Support):** WISMO tickets up 40% after the Delhivery delays; approved a ₹2,500 credit for a missed BharatFleet sofa slot.
- **Rohan (Marketing):** Diwali creator partnerships; **Ghar Club early access to the Wedding Edit moved from 14 to 12 November**.
- **Kabir (Merchandising):** sheesham sofa back in stock on 2 December; Noor Metalcraft credit note for the damaged lamps.
- **Anand (CFO, restricted):** board wants the Diwali discount kept at 20% to protect the 45% gross margin.
- **Farah (People):** two open roles; Arjun is the hiring manager.

### What maintenance already did

Newer notes mark older document sections as **updated** or **partly
updated**: the Shipping Policy (free shipping threshold), the Diwali Campaign
key dates (Wedding Edit early access), the Assortment stock risks (sofa back
in stock), and one of Ananya's two notes (a duplicate).

### What the root learned

A handful of **anonymized lessons** from Desi Living, for example about
lowering the free-shipping threshold during the festive peak. No names,
numbers or company details.

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

- **Pick:** Neha (Support agent)
- **Ask:** *Can I give a customer a ₹2,000 credit without asking anyone?*
- **Expect:** Green badge **"Answered from your department"**: no, agents can give up to ₹500; a team lead can approve up to ₹2,000. Click **sources** to see the Support Playbook.
- **Say:** "Neha's question is answered from her own department's knowledge, with the source it came from."

### F2. Falling back to the company brain

- **Pick:** Neha
- **Ask:** *How many days of earned leave do I get?*
- **Expect:** Orange badge **"Answered from your company"**: 18 days a financial year, up to 10 can be carried forward.
- **Say:** "Support doesn't know about leave, so the company brain answered. It goes up one level, never sideways into other companies."

### F3. Conversation memory

- **Pick:** Neha, right after F2
- **Ask:** *and can I carry any over?*
- **Expect:** An answer about the 10 carry-forward days, with **"Understood as: How many days of earned leave can I carry over…"** under it.
- **Say:** "It remembers the conversation, so follow-up questions work like they would with a colleague."
- Use **New conversation** to start fresh.

### F4. Permissions: restricted data

- **Pick:** Tanvi (Finance analyst). **Ask:** *What was Q2 revenue?*
- **Expect:** **"Not found in your brains."**
- **Pick:** Anand (CFO). **Ask** the same.
- **Expect:** **₹11.9 crore**, up 21% year over year, from the Q2 FY27 Results PDF (gross margin 47%).
- **Say:** "Same department, same question, different answers. Tanvi isn't cleared for restricted finance data; the CFO is."

### F5. Permissions: access between departments

- **Pick:** Neha. **Ask:** *What's the free shipping threshold right now?*
- **Expect:** **₹699** from Dhanteras to 15 November, reduced from ₹999.
- **Pick:** Sana (Marketing). **Ask** the same.
- **Expect:** **₹999**.
- **Say:** "The ₹699 change is a note in Operations. Neha has permission to read Operations, so she gets the update. Sana doesn't, so she still gets the official policy. Nobody sees data they aren't allowed to, and nobody loses information because of a note they can't read."

### F6. Isolation between companies

- **Pick:** Karan (BharatFleet). **Ask:** *How many days of earned leave do we get?*
- **Expect:** **15 days**, never Desi Living's 18.
- **Ask** as Karan: *Who is Ananya Reddy?* → **Not found**: she is Desi Living's customer.
- **Say:** "Each company's knowledge is completely separate, even between partners who work together every day."

### F7. PDF and Word documents

- **Pick:** Sana. **Ask:** *What is the hotel cap for a manager?*
- **Expect:** **₹6,500 per night**. This comes from a **table inside the Word handbook**.
- **Pick:** Neha. **Ask:** *Do customers need to send back a damaged item?*
- **Expect:** Not for items under ₹3,000; a photo or unboxing video is enough. This comes from the **Returns Policy PDF**.
- **Say:** "It reads the documents companies actually have: PDF and Word, including tables."

### F8. Remembering new information

- **Pick:** Arjun (Head of Support), open the **Remember** tab.
- Type each of these and click **Remember**:
  - *Met the Kaveri Stores buyer today, they want delivery before 10 December* → shows **episodic** (an event)
  - *Our standard payment terms for wholesale orders are net 45* → shows **semantic** (a fact)
  - *The Kaveri Stores buyer prefers WhatsApp over email* → shows **user** (about a person)
- Click **Save to long-term**.
- Back on **Ask**: *When does Kaveri Stores need delivery?* → **before 10 December**.
- **Pick:** Rohan (Marketing). Ask the same → **Not found**: Arjun's notes stay in Support.
- **Say:** "New knowledge goes into short-term memory, gets sorted automatically, and is saved to long-term memory, only visible to the right people."

### F9. Memory keeps itself up to date (live)

- **Pick:** Arjun. **Remember:** *From today support agents can give goodwill credits up to ₹1,000 without approval*. Click **Save to long-term**.
- Wait about **10 seconds** (maintenance runs in the background).
- **Pick:** Neha. **Ask:** *How much credit can I give without approval?*
- **Expect:** **₹1,000**, mentioning it was raised from ₹500.
- Already loaded: **Rohan**, *When does Ghar Club early access to the Wedding Edit start?* → **12 November**, moved from the plan's 14 November.
- **Say:** "When something changes, the brain notices that the new note updates an older fact and uses the newer one. Nothing is deleted; you can still see the history in the graph."

### F10. Graph: what each person can see

- Open **http://localhost:8765/graph-view**, pick **Anand (CFO)**.
- **Expect:** All six department zones filled, restricted memories with a **red outline**.
- Pick **Sana**. **Expect:** Most zones empty; she only sees Marketing and company knowledge.
- Click a node (e.g. a department or a person) to see its details and relationships.
- **Say:** "This is the same permission model, made visible. The graph is built by the database for this person, so it can't show more than they're allowed to see."

### F11. Graph: the whole SecondBrain

- Pick **★ Whole SecondBrain (admin)**, enter the admin token, click **Open**.
- **Expect:** The gold **SecondBrain Root** in the middle with its **learned lessons** around it; Desi Living and BharatFleet as clusters.
- Faded nodes with an orange or yellow dashed ring are **updated or partly updated** memories; click one to see the **"Partly updated by"** link and the reason.
- **Say:** "This is our view of the whole network. The root holds only anonymous lessons learned from companies that agreed to share."

### F12. Learning at the root

- In the whole view, click any small node right around the root to read a lesson.
- To run learning again (it skips lessons it already has):

  ```bash
  curl -s -X POST http://localhost:8765/admin/learn -H "X-Admin-Token: $(grep '^SB_ADMIN_TOKEN=' .env | cut -d= -f2)"
  ```

- **Say:** "Lessons are written without any names or numbers, then checked by code; anything identifying is rejected. Restricted data is never used, and BharatFleet, which doesn't share, contributes nothing. Company users never get answers from the root."

### F13. Onboarding a new company live

1. Make a folder `~/Desktop/nova_demo` with a file `company.json`:

   ```json
   {
     "id": "nova",
     "name": "Nova Tech",
     "sharing": "none",
     "departments": [{"id": "eng", "name": "Engineering"}],
     "users": [{"id": "riya", "name": "Riya Raman (Engineer)", "department": "eng", "role": "member"}]
   }
   ```

   and a file `oncall.md`:

   ```
   # On-call Guide

   ## Rotation
   The on-call rotation changes every Monday at 10am IST. P1 incidents are escalated to the engineering manager within 15 minutes.
   ```

2. Load them:

   ```bash
   uv run python -m secondbrain.admin onboard ~/Desktop/nova_demo/company.json
   uv run python -m secondbrain.admin import root.nova.eng ~/Desktop/nova_demo/oncall.md --sensitivity internal
   ```

3. **Refresh** the chat. Pick **Riya**, ask *When does on-call change?* → **Monday at 10am**.
4. **Say:** "A new company is live in minutes: one file for its structure, and its documents."

### F14. Proof that the database enforces permissions

For technical audiences. This connects the way the app does and asks for all memories:

```bash
docker compose exec postgres psql "postgresql://secondbrain_app:dev_app_password@localhost/secondbrain" -c "SELECT count(*) FROM memories;"
```

Result: **0**, because no person is set.

```bash
docker compose exec postgres psql "postgresql://secondbrain_app:dev_app_password@localhost/secondbrain" -c "BEGIN; SELECT set_config('app.user_id','sana',true); SELECT DISTINCT brain_path FROM memories; COMMIT;"
```

Result: only `root.desiliving` and `root.desiliving.marketing`.

- **Say:** "Even if someone bypassed our application, the database itself would only return what that person may see."

### F15. Who knows what (relationships)

When documents and notes are saved, the AI picks out the people, customers,
suppliers, products, carriers, places and topics they mention, and how they
relate. These are kept in PostgreSQL and in a **FalkorDB** graph per company.

- **Setup (once, after loading the data):**

  ```bash
  uv run python -m secondbrain.relationships
  ```

  It prints the number of entities per company. New notes saved with
  **Save to long-term** are added by themselves.

- **Try it** in the Explore page (Who knows what), or in a terminal as
  different people (`X-User-Id` picks the person):

  ```bash
  curl -s --get localhost:8765/experts --data-urlencode "topic=Diwali" -H "X-User-Id: meera" | python3 -m json.tool
  curl -s localhost:8765/entities/BharatFleet%20Logistics -H "X-User-Id: neha" | python3 -m json.tool
  ```

| # | Person | Ask | Expect |
|---|---|---|---|
| 1 | Neha | `/experts?topic=Ananya Reddy` | **Neha Kulkarni** (wrote her notes), in **Customer Support** |
| 2 | Sana | `/experts?topic=Ananya Reddy` | **Nobody**: the Ananya notes are in Support, which Sana can't read |
| 3 | Meera (CEO) | `/experts?topic=brass floor lamp` | **Priya Nair** and **Kabir Singh**; Operations and Merchandising |
| 4 | Meera | `/experts?topic=Diwali` | **Rohan Kapoor** and **Anand Krishnan**, among others; Marketing and Finance |
| 5 | Neha | `/experts?topic=shipping` | **Vikram Rao** (his ₹699 free shipping note); memories across the company, Operations and Support |
| 6 | Sana | `/experts?topic=shipping` | **No people**: she can't see Vikram's Operations note, the same rule as F5 |
| 7 | Kabir | `/entities/Noor Metalcraft` | **Supplier** in Moradabad, credit note for the 36 damaged **brass floor lamps** |
| 8 | Meera | `/entities/BharatFleet Logistics` | **Carrier**, delivers the sofas and side tables, including the restricted rate contract |
| 9 | Neha | `/entities/BharatFleet Logistics` | **Fewer memories**: the restricted Carrier Contracts are left out |
| 10 | Rohan | `/entities/sheesham sofa` | **Product**, **Saharanpur Woodcraft** makes it, back in stock on 2 December |
| 11 | Sana | `/entities/Saharanpur Woodcraft` | Only what the company overview says: Merchandising's notes are hidden from her |
| 12 | Tanvi | `/entities/gross margin` | **404**: only in restricted Finance data |
| 13 | Anand | `/entities/gross margin` | Memories in Finance: Q2 results and the board's 45% floor |
| 14 | Karan (BharatFleet) | `/entities/Noor Metalcraft` | **404**: Desi Living's graph is a different graph |
| 15 | Meera | `/entities/Vikram Rao` | **Person** (`person_id: vikram`), prefers updates in **#ops-alerts**; Priya wrote that note |

The AI's extraction varies a little between runs, so names and counts may
differ slightly; the permission results (2, 6, 9, 12, 14) never do.

- **Say:** "Ask who knows about a customer, a supplier or a topic, and it points
  you to the people and teams. The relationship graph has no permissions of
  its own, so we never trust it on its own: every graph question is limited to
  the memories PostgreSQL says this person may read. That's why Sana gets
  nobody for Ananya Reddy, and Tanvi gets nothing on gross margin."

### F16. Automated tests

```bash
uv run pytest
```

Expect **53 passed**, including tests that no entity, person or relationship leaks across permissions or companies. They run on a separate test database.

---

## 5. Questions clients usually ask

| Question | Answer |
|---|---|
| Where is our data stored? | In our PostgreSQL database, separated per company, with access enforced by the database. It can also run on your own servers. |
| Who can see what? | You define departments and roles; each person sees their department, the company level, and anything they're explicitly granted. Show F4, F5, F10. |
| Does our data train the AI? | No. Your documents are stored as memory; the AI reads them to answer, but no model is trained on them. |
| Does our data go to other companies? | Never. Only if you opt in, anonymous lessons with no names or numbers go to our root, and other companies never query it directly. |
| What happens when information changes? | Newer notes update older facts automatically (F9). Nothing is deleted, so there's a full history. |
| Can it tell me who knows about something? | Yes: ask who knows about a customer, supplier or topic, or what we know about a product (F15). It only counts what you may read. |
| Which documents does it read? | PDF, Word, Markdown and text today. Google Drive, Slack and email connectors are planned. |
| How do people log in? | Today the demo uses a user picker. In production it uses your existing sign-in (Google, Microsoft, Okta). |

---

## 6. Know before the demo

- **The VPN must stay on.** Without it, questions fail.
- **FalkorDB must be running** (`docker compose up -d --wait`) for F15. If it was restarted, the first query rebuilds the graph from PostgreSQL by itself.
- **Answers take 2 to 5 seconds**, longer for follow-ups and the first question after starting.
- **Reset the data** before a client demo so practice notes are gone.
- The **user picker is a demo stand-in** for real sign-in; mention SSO is planned.
- The **whole-SecondBrain view** shows every company's data. With a real client, only show it with sample companies, never another client's data.
- **Maintenance isn't perfect.** It can mark a section as "partly updated" when it only added detail. That's harmless: nothing gets hidden in that case.
- Only **load real client documents** if the client agrees to them being processed by the AI models behind our LiteLLM proxy.
