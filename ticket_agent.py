# ticket_agent.py
from crewai import Agent, Task, Crew, Process, LLM # crew = members, process = ablauf
# from langchain_ollama import ChatOllama crewai hat es schon integriet 
import pandas as pd
from datetime import datetime # zeitstempel für berrichte zu erstellen 
import os

# LLM Setup (konsistent & deterministischer)
ollama_llm = LLM(
    model="ollama/llama3.2",
    temperature=0.2,      # niedriger für Klassifikation & Entscheidungen
    base_url='http://localhost:11434'
#    config={'num_ctx': 8192} hier auch in crewai schon standsiert wenn man trotz machen will, je mehr wert desto mehr v Ram  
)

# === Agents ===
classifier = Agent(
    role="Senior Ticket Classifier & Prioritizer",
    goal="Tickets präzise kategorisieren, priorisieren und relevante Tags vergeben", 
    backstory="""Du bist ein erfahrener Support-Lead mit 12 Jahren Erfahrung in einem SaaS-Unternehmen. 
    Du kennst alle gängigen Ticket-Kategorien und weißt genau, wann etwas dringend ist.""", # log in > quittung frage
    llm= ollama_llm,   
    verbose=True, #in Terminal zeigt was er gerade denkt
    allow_delegation=False # allow_delegation = andere Agents ?
)

responder = Agent(
    role="Customer Support Specialist",
    goal="Professionelle, empathische und lösungsorientierte Antworten erstellen",
    backstory="""Du bist ein sehr guter Support-Mitarbeiter, der immer freundlich bleibt, klare Schritte gibt und bei Bedarf auf Escalation hinweist.""",
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False
)

manager = Agent(
    role="Support Operations Manager",
    goal="Qualität sicherstellen und effiziente Escalation-Entscheidungen treffen",
    backstory="""Du entscheidest datenbasiert, ob ein Ticket automatisch beantwortet werden kann oder menschliche Intervention nötig ist. 
    Du achtest auf Risiko, Komplexität und Kundenzufriedenheit.""",
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False
)

# === Tasks ===
def create_tasks(ticket: dict): # dict = key vlaue pairs
    classify_task = Task(
        description=f"""
        Analysiere folgendes Support-Ticket und liefere eine strukturierte Klassifikation:

        Ticket ID: {ticket.get('id', 'N/A')}
        Subject: {ticket.get('subject', '')}
        Description: {ticket.get('description', '')}

        Gib zurück:
        - Kategorie (Technik, Rechnung, Zugang, Feature-Request, Sonstiges)
        - Priorität (Hoch, Mittel, Niedrig)
        - Tags (max 3, z.B. #billing, #login, #urgent)
        - Kurze Begründung
        """,
        agent=classifier,
        expected_output="JSON-ähnliche strukturierte Ausgabe mit Kategorie, Priorität, Tags und Begründung"
    )

    respond_task = Task(
        description="Erstelle eine vollständige, professionelle und hilfreiche Antwort-E-Mail auf das Ticket. Nutze die Klassifikation aus der vorherigen Task.",
        agent=responder,
        expected_output="Vollständige E-Mail (Betreff + Anrede + Inhalt + Grußformel)",
        context=[classify_task] # jetzt weis responder kategotie prioritäts ...
    )

    decide_task = Task(
        description="Entscheide basierend auf Klassifikation und Antwort-Entwurf: Kann das Ticket automatisch beantwortet werden oder muss es eskaliert werden? Begründe deine Entscheidung.",
        agent=manager,
        expected_output="Entscheidung (Auto-Answer / Escalate) + klare Begründung + ggf. Escalation-Grund",
        context=[classify_task, respond_task]
    )

    return [classify_task, respond_task, decide_task]

# Hauptfunktion
def process_ticket(ticket_data: dict):
    tasks = create_tasks(ticket_data)
    
    crew = Crew(
        agents=[classifier, responder, manager],
        tasks=tasks,
        process=Process.sequential,   # wichtig: Reihenfolge einhalten
        verbose=True, # 1 true (wichtigsten Gedankengänge), 2 false, Hier 2 zeigt alle schritte (Vollbild-Modus)
        memory=False # lokalen Vektordatenbank im Hintergrund ab
    )
    
    result = crew.kickoff()
    return result

# Beispiel-Tickets zum Testen
if __name__ == "__main__": # nur beim play anklick durchgeführt nicht beim import in andere skript
    test_tickets = [
        {
            "id": "T001",
            "subject": "Login funktioniert nicht",
            "description": "Seit dem Update kann ich mich nicht mehr einloggen. Bekomme immer Fehler 403."
        },
        # {
        #     "id": "T002",
        #     "subject": "Rechnung falsch abgerechnet",
        #     "description": "Mir wurde der doppelte Betrag abgebucht. Das muss dringend korrigiert werden!"
        # },

#         {
#     "id": "T003",
#     "subject": "Login funktioniert nicht - KRITISCH",
#     "description": "Seit dem Update kann ich mich nicht mehr einloggen. Bekomme immer Fehler 403. Das blockiert unsere gesamte Buchhaltung, wir können seit Stunden nicht arbeiten! Wenn das nicht bis heute Abend gelöst ist, übergeben wir das unserem Anwalt wegen Schadensersatz!"
# }
    ]
    
    for ticket in test_tickets:
        print(f"\n{'='*60}")
        print(f"Verarbeite Ticket {ticket['id']}")
        print(f"{'='*60}")
        result = process_ticket(ticket)
        print(result)