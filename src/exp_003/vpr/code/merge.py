import glob
import os

estensioni = ['*.py', '*.yaml']
nome_file_output = "raccolta_completa.txt"
script_corrente = os.path.basename(__file__)
separatore = "-" * 40

with open(nome_file_output, "w", encoding="utf-8") as outfile:
    for estensione in estensioni:
        for nome_file in glob.glob(estensione):
            if nome_file == script_corrente:
                continue
            
            if not os.path.isfile(nome_file):
                continue

            outfile.write(f"{nome_file}:\n")
            outfile.write(f"{separatore}\n")
            
            with open(nome_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
            
            outfile.write("\n\n")