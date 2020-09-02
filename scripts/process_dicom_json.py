import json

with open("/mnt/data/LungNodules/besser-list-complete_metadata-orig.json", "r") as f_in:
    old_metadata = json.load(f_in)

with open("/mnt/data/LungNodules/dicom_table.json", "r") as f_in:
    dicom_table = json.load(f_in)

new_metadata = {}
for patient in old_metadata.keys():
    for series in old_metadata[patient].keys():
        patient_id = patient+"_ser_"+series
        old_patient_metadata = json.loads(old_metadata[patient][series])
        new_patient_metadata = {}
        for key, data in old_patient_metadata.items():
            if key in dicom_table.keys() and "Value" in data.keys():
                value = data["Value"]
                if len(value) == 1:
                    value = value[0]
                tag = dicom_table[key]
                new_patient_metadata[tag] = value
        new_metadata[patient_id] = new_patient_metadata

with open("besser-list-complete_metadata.json", "w") as f_out:
    json.dump(new_metadata, f_out)
