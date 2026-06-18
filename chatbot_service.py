import os, re, uuid, datetime
from openai import OpenAI
from database.db import save_chat_message, get_chat_history, get_user, get_consultations, get_bookings

# Attempt to load key from environment, fallback to user's provided key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = """You are MediPredict AI, a highly intelligent and empathetic medical assistant chatbot.
You are part of the MediPredict multimodal application which diagnoses diseases and analyzes medical reports.
Your main job is to:
1. Listen to the user's symptoms and help diagnose potential conditions.
2. Analyze any medical report summaries provided to you.
3. Assist the user in finding doctors and booking appointments. If they provide a location or specialty, use the provided doctor context to recommend real doctors available on this platform.
4. If the user explicitly asks to book a doctor, ask them to pick exactly one of the available slots for that doctor. Once they confirm a specific slot, you MUST conclude your message directly by including this exact XML block at the very end:
   <BOOK_APPOINTMENT doctor="[Doctor Name]" specialty="[Specialty]" hospital="[Hospital/Location]" slot="[Slot Time]">
   DO NOT use this tag until the user has confirmed a specific slot!
5. Be professional, supportive, and succinct. Limit responses to a few paragraphs at most unless requested otherwise.
6. Always include a disclaimer that you are an AI and not a substitute for a licensed doctor for serious emergencies.

When a user asks what you can do, let them know you can analyze their symptoms in real-time, read medical reports, and officially book doctors in their area directly through this chat.
"""

import json
ALL_DOCTORS = []
try:
    with open(os.path.join(os.path.dirname(__file__), "data", "india_doctors_dataset.json"), "r", encoding="utf-8") as f:
        ALL_DOCTORS = json.load(f)
except Exception:
    pass

def find_relevant_doctors(query, latest_disease=None):
    query_clean = ''.join(e for e in query.lower() if e.isalnum())
    
    candidate_doctors = ALL_DOCTORS
    if latest_disease:
        try:
            from app import match_doctors_for_disease
            matched = match_doctors_for_disease(latest_disease)
            if matched:
                candidate_doctors = matched
        except Exception:
            pass

    matches = []
    # simple heuristic: if a doctor's district, state, or specialty is in the query
    for d in candidate_doctors:
        d_dist = ''.join(e for e in d.get("district", "").lower() if e.isalnum())
        d_state = ''.join(e for e in d.get("state", "").lower() if e.isalnum())
        d_spec = ''.join(e for e in d.get("specialty", "").lower() if e.isalnum())
        
        # e.g., mapping trichy to tiruchirappalli
        if "trichy" in query_clean and "tiruch" in d_dist:
            matches.append(d)
            continue
            
        if (d_dist and d_dist in query_clean) or \
           (d_state and d_state in query_clean) or \
           (d_spec and d_spec in query_clean):
            matches.append(d)
            
    # If a specific disease was found but no location matches perfectly from the query, 
    # just return the top doctors for that disease anyway.
    if latest_disease and not matches:
        return candidate_doctors[:5]
        
    return matches[:5] # limit to top 5 to save context


def generate_chat_response(user_id, user_message, extracted_report_text=None):
    """
    Process the user's chat message, retrieving context, calling Groq Llama 3, and saving the outcome.
    """
    # 1. Fetch user data for context to give the Chatbot full DB access
    user_data = get_user(user_id)
    user_name = user_data["full_name"] if user_data else "User"
    user_consults = get_consultations(user_id)
    user_books = get_bookings(user_id)

    db_context_lines = []
    if user_data:
        db_context_lines.append(f"User Profile: Age/DOB: {user_data.get('dob','Unknown')}, Blood Group: {user_data.get('blood_group','Unknown')}, Allergies: {user_data.get('allergies','None')}")
    
    if user_consults:
        db_context_lines.append(f"\nPast Diagnoses/Consultations ({len(user_consults)} total):")
        for c in user_consults[:5]: # Send top 5 most recent
            db_context_lines.append(f"- {c.get('date')}: Diagnosed with '{c.get('primary_diagnosis')}' (Severity: {c.get('severity')}) based on symptoms: '{c.get('symptom_text')}'. Recommendations: {c.get('recommendations')}")
    
    if user_books:
        db_context_lines.append(f"\nDoctor Appointments ({len(user_books)} total):")
        for b in user_books[:5]: # Send top 5 most recent
            db_context_lines.append(f"- Ref {b.get('booking_ref')}: Appt with Dr. {b.get('doctor_name')} ({b.get('specialty')}) at {b.get('hospital')} on {b.get('date')} slot {b.get('slot')}. Status: {b.get('status')}")

    db_context = "\n".join(db_context_lines)

    # 2. Add contextual report text to the message if uploaded
    if extracted_report_text:
        user_message += f"\n\n[USER ATTACHED REPORT CONTENT]:\n{extracted_report_text[:3000]}"

    # Add relevant doctor context if the user is asking about booking or doctors
    if any(w in user_message.lower() for w in ["book", "doctor", "appointment", "near", "find", "who", "tamilnadu"]):
        latest_disease = user_consults[0].get("primary_diagnosis") if user_consults else None
        rel_docs = find_relevant_doctors(user_message, latest_disease)
        if rel_docs:
            doc_context = "\n".join([f"- Dr. {d['name']} ({d.get('specialty')}) in {d.get('district')}, {d.get('state')} (Rating: {d.get('rating')}). Slots: {d.get('slots')}" for d in rel_docs])
            user_message += f"\n\n[SYSTEM NOTE: The following doctors ideally matching the user's location query or recent disease ({latest_disease}) are available on the platform:\n{doc_context}\nPlease suggest these doctors to the user and tell them they can finalize the booking on the 'Book Doctor' page.]"


    # 3. Save the new user message to DB
    save_chat_message(user_id, "user", user_message)

    # 4. Retrieve conversation history to maintain context
    history_records = get_chat_history(user_id, limit=20)
    
    # 5. Build messages array for Groq
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(user_name=user_name)},
        {"role": "system", "content": f"DATABASE CONTEXT FOR {user_name.upper()}:\n{db_context}\n\nUse this data to answer questions about the user's past medical history, upcoming appointments, allergies, or blood type perfectly."}
    ]

    
    # Append past history
    for record in history_records[-15:]:  # keeping last 15 messages for context length limits
        messages.append({
            "role": record["role"],
            "content": record["content"]
        })

    # The latest user message is already in `history_records` since we saved it above, 
    # but we will just pass the constructed `messages` list directly to the LLM.

    # 6. Call Groq
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
            top_p=0.9,
            stream=False
        )
        assistant_response = completion.choices[0].message.content.strip()
    except Exception as e:
        assistant_response = f"I apologize, but I am currently facing a connection issue with my AI core. Error: {str(e)}"
    
    # Check for <BOOK_APPOINTMENT ...> action
    match = re.search(r'<BOOK_APPOINTMENT\s+doctor="([^"]+)"\s+specialty="([^"]*)"\s+hospital="([^"]*)"\s+slot="([^"]+)">', assistant_response)
    if match:
        doc_name, specialty, hospital, slot = match.groups()
        ref = f"MP{uuid.uuid4().hex[:6].upper()}"
        
        from database.db import save_booking
        booking_data = {
            "ref": ref, 
            "doctor": doc_name.replace("Dr. ", ""),  # clean
            "specialty": specialty, 
            "hospital": hospital,
            "slot": slot, 
            "date": datetime.datetime.now().strftime("%d %B %Y"), 
            "color": "#0EA5E9"
        }
        
        # Save to DB officially
        save_booking(user_id, booking_data)
        
        # Strip the XML from user view and append success checkmark
        assistant_response = assistant_response[:match.start()] + assistant_response[match.end():]
        assistant_response += f"\n\n--- \n✅ **Success!** Your appointment with **Dr. {booking_data['doctor']}** at **{slot}** has been securely booked. \nReference ID: `{ref}`."

    # 7. Save the assistant response
    save_chat_message(user_id, "assistant", assistant_response)
    
    return assistant_response
