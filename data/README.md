# Daten / Data

Die Originaldaten (Fahrgastströme, Wetter, Events, Sperrungen, Energieverbrauch, Stationen)
wurden im Rahmen des InnoTrans Hackathons 2026 von Alstom bereitgestellt und dürfen nicht
weitergegeben werden. Sie sind deshalb nicht in diesem Repository enthalten.

Auch die abgeleiteten Netzdaten (VBB-GTFS-Extrakt, BVG-Linienbänder und die daraus erzeugten
Graph-Dateien) fehlen. Die Skripte zum Erzeugen des Netzgraphen liegen unter `src/graph_db/`.

Für eigene Tests kann `sbahn_extra_EXAMPLE.csv` im Repository-Root als Vorlage für das
Format zusätzlicher Stationsdaten genutzt werden.

---

*The original dataset was provided by Alstom for the InnoTrans Hackathon 2026 and may not be
redistributed, so it is not included here. Derived network data (VBB GTFS extract, BVG line
PDFs and the generated graph files) is not included either; the scripts that build the
network graph are in `src/graph_db/`.*
