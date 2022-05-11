import spacy
import json

nlp=spacy.load('en_core_web_sm')
mapping ={
    "LOC": "I-LOC",
    "GPE": "I-LOC",
    "PERSON": "I-PER",
    "ORG": "I-ORG",
    "MONEY": "I-MISC",
    "DATE": "I-MISC",
    "TIME": "I-MISC"
}
def show_ents(doc):
    doc = nlp(doc.get("data","Default Sentence"))
    ret_list=[]
    cnt = 0
    if doc.ents:
        for ent in doc.ents:
            cnt+=1
            ret_list.append([ent.text, mapping[ent.label_]])
            # print(ent.text+'-',str(ent.start_char)+'-'+str(ent.end_char)+'-'+ent.label_+'-'+str(spacy.explain(ent.label_)))
    else:
        print('No entities found')

    return json.dumps({"count":cnt,"data":ret_list})