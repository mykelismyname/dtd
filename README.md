# dtd_decoding

Detecting signals of Difficult to Treat or Treatment Resistant Depression

### Problem
Categorical diagnosis, 
1. Don't capture the dimensional nature of mental illnesses --  Two patients who have ICD codes for e.g. "Depression" or "Paranoid Schizophrenia" can look very different in terms of which clinical features (symptoms, signs) are present and at what level of severity 
2. Are rarely granular enough to 'map' onto interventions appropriate for a given patient's specific presentation
3. On their own are not sufficient tracing patients eligible for clinical services or trials
 
### Work around

- Develop a chain of feature-specific agents or detectors, that either **Support**, **Refute** or **Decline presence of any signalling information** regarding

a patient's mental illness status that is indicative of DTD.
