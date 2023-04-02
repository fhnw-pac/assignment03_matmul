# Assignment03 MatMul

Punkte werden wie folgt vergeben: 

Code 
- ½P für die korrekte Verwendung der CUDA API 

Implementierungen 
- ½P für eine CUDA basierte Matrix-Multiplikation, welche hauptsächlich das Global-Memory benutzt 
- 2P für eine CUDA basierte Version, welche das Shared-Memory korrekt benutzt und so tendenziell schneller ist als vorherige Version. 
-2P für eine CUDA basierte Version, welche tendenziell noch schneller/optimierter ist als die vorherige Shared-Memory Version. 
    oder 
    2P für eine optimierte CPU Version, welche gegen die Implementierung von cuBLAS antritt.  

Dokumentation 
- ½P für die sinnvolle Untersuchungen/Messungen mit den CUDA Profiling Tools und deren Analyse. Ein sauberes Zeit-Profiling aller gewählten Ansätze (ebenso vs CPU Code). Unterscheiden Sie zwischen Init-Aufwand und der wiederholten Ausführung des Codes/der Kernel. 
- ½P für eine saubere Dokumentation der verwendeten Ansätze und des allgemeinen Aufbaus des Codes. Sowie sinnvollen Schlussfolgerungen der erarbeiteten Resultate. 
