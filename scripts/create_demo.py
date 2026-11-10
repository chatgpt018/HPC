"""Generate clearly synthetic practice reports, never market-fact evidence."""
from __future__ import annotations

import csv
from pathlib import Path

SAMPLES = [
('energy','Solar panel supply chains','Research on solar modules, silicon wafer production, installation capacity, renewable energy and utility procurement.'),
('energy','Battery value chains','Research on EV batteries, lithium supply chains, battery recycling, electric vehicles and storage technologies.'),
('energy','Grid investment','Research on grid capacity, transmission infrastructure, renewable integration, battery storage and electricity demand.'),
('healthcare','Clinical research','Research on pharmaceutical clinical trials, drug development, regulatory approval and medical evidence.'),
('healthcare','Hospital systems','Research on hospital staffing, patient scheduling, medical devices, healthcare operations and diagnostics.'),
('healthcare','Biopharmaceuticals','Research on biotechnology manufacturing, therapeutics, pharmaceutical production and clinical trials.'),
('technology','Semiconductor manufacturing','Research on semiconductor fabrication, chip packaging, process nodes, materials and industrial demand.'),
('technology','AI datacenters','Research on artificial intelligence infrastructure, chips, datacenters, networking systems and compute demand.'),
('technology','Enterprise software','Research on cloud platforms, enterprise software, subscription services, cybersecurity and data systems.'),
]


def main():
    folder=Path('examples/synthetic_reports')
    folder.mkdir(parents=True,exist_ok=True)
    for i,(sector,title,text) in enumerate(SAMPLES,1):
        path=folder/f'synthetic_{i:02d}_{sector}.csv'
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['title','category','description','disclaimer'])
            writer.writeheader()
            writer.writerow({'title':title,'category':sector,'description':text,
                             'disclaimer':'SYNTHETIC EDUCATIONAL TEXT. Not a real market report or financial forecast.'})
    print(f'Created {len(SAMPLES)} labeled synthetic CSV practice documents in {folder}')

if __name__=='__main__':
    main()
