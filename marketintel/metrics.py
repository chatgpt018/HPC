"""Source-traceable metrics from uploaded CSV files, not text guesses or fabricated values."""
from __future__ import annotations

import csv
from collections import defaultdict

from .storage import records

REQUIRED = {'market','year','value','unit','region','currency'}


def load_market_series() -> tuple[list[dict],list[str]]:
    """Read only CSVs whose columns explicitly declare a comparable market time series."""
    points=[]; warnings=[]
    for document in records():
        if document['file_type']!='.csv':
            continue
        filename=document['local_path']
        try:
            with open(filename,'r',encoding='utf-8-sig',newline='') as handle:
                reader=csv.DictReader(handle)
                if not reader.fieldnames or not REQUIRED.issubset(set(reader.fieldnames)):
                    continue  # Non-metrics CSV; it can still be a document.
                for number,row in enumerate(reader,start=2):
                    try:
                        year=int(row['year'])
                        value=float(row['value'])
                        if not 1900<=year<=2200 or not 0<value<float('inf'):
                            raise ValueError('Invalid year or positive finite value')
                        market=row['market'].strip(); unit=row['unit'].strip(); region=row['region'].strip(); currency=row['currency'].strip()
                        if not all([market,unit,region,currency]):
                            raise ValueError('Missing required label')
                        points.append({'market':market,'year':year,'value':value,'unit':unit,
                                       'region':region,'currency':currency,
                                       'document_id':document['doc_id'],'filename':document['filename'],
                                       'row':number})
                    except (ValueError,TypeError,KeyError) as exc:
                        warnings.append(f"{document['filename']} row {number}: {exc}")
        except (OSError, UnicodeError) as exc:
            warnings.append(f"{document['filename']}: cannot read metrics: {exc}")
    return points,warnings


def growth_table() -> tuple[list[dict],list[str]]:
    points,warnings=load_market_series()
    groups=defaultdict(list)
    for point in points:
        key=(point['market'],point['region'],point['unit'],point['currency'])
        groups[key].append(point)
    rows=[]
    for key,items in groups.items():
        seen={}
        for item in items:
            if item['year'] in seen:
                warnings.append(f'Conflicting duplicate market-year data: {key}, {item["year"]}; series omitted')
                break
            seen[item['year']]=item
        else:
            if len(items)<2:
                continue
            items=sorted(items,key=lambda v:v['year'])
            a,b=items[0],items[-1]
            years=b['year']-a['year']
            cagr=(b['value']/a['value'])**(1/years)-1 if years>0 else None
            for prev,now in zip(items,items[1:]):
                rows.append({'market':key[0],'region':key[1], 'unit':key[2],'currency':key[3],
                             'year':now['year'],'value':now['value'],
                             'yoy_percent':100*(now['value']/prev['value']-1) if now['year']==prev['year']+1 else None,
                             'series_cagr_percent':100*cagr if cagr is not None else None,
                             'source':f"{now['filename']} row {now['row']}",
                             'start_year':a['year'],'end_year':b['year']})
    return rows,warnings
