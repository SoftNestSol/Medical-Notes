Important notes:

What do we want? We want to make an application that integrates an AI Medical Notes module.

For that, we looked into MediQA paper for the 2023 shared task and on the Empirical Study on patient doctor paper from Microsoft, 2023 too - Ben Abacha

(Next we should describe what these people found and what their results were)

Our own dataset: 30 recordings and the corresponding patient files from real patients (we received annomized versions)

STT Layer - Our aproach in the app. 

STT Models: The easy part; how do they plug into our app 

Dummy aproach: Do what they did and validate on our own data set, as a blind try

Didnt work out too good. Tried a couple more models.

Added to MTS Dialog our dataset + distilled versions of the convos and even more convos. Niche on the phisiotheraphy work 

Tried to make medical notes + evaluation of the results 


### Claude info directions
Goal

Generate clinical note sections from Romanian doctor-patient dialogues (ASR → translated → FLAN-T5-base fine-tuned on MTS-Dialog)
Failures observed

Negation flips (C7 vs C5-C6, "no symptomatology")
Doctor's reasoning attributed to patient
Examiner maneuvers described as patient actions
Numeric values fused incorrectly (7/10 pain + 3-4 days/week → "7 times a week")
Hallucinated symptoms from leading questions
Why

Distribution shift: trained on US English, tested on Romanian-translated clinical speech
250M params too small for reliable negation/scope handling
ASR noise on medical terminology
Translation artifacts
Next steps

Measure ROUGE on MTS-Dialog test split to confirm fine-tune worked
Measure ASR word error rate
Collect more in-domain Romanian pairs
Try input normalization with a larger model as preprocessing
Scale up model once enough data exists