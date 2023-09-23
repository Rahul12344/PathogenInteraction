from collections import defaultdict

class Labels:
    def __init__(self, export_file_path, VIP_file_path):
        self.VIP_file = VIP_file_path
        self.export_file = export_file_path
    
    def BuildGenomeMap(self):
        export_file = open(self.export_file, "r")
        lines = export_file.readlines()
        self.map = defaultdict(str)
        for line in lines:
            ensemble = line.split("\t")
            #print(ensemble)
            if(ensemble[0][0:4] == "ENSG"):
                if ensemble[1].rstrip("\n") != "":
                    self.map[ensemble[0]] = ensemble[1].rstrip("\n")
                else:
                    self.map[ensemble[0]] = ensemble[0]
        export_file.close()
    
    def BuildLabeler(self):
        self.BuildGenomeMap()
        
        self.labels = defaultdict(int)
        
        VIP_file = open(self.VIP_file, "r")
        lines = VIP_file.readlines()
    
        for line in lines:
            ensemble = line.split("\t")
            if(ensemble[0][0:4] == "ENSG"):
                if ensemble[0] not in self.map:
                    self.labels[ensemble[0]] = 1
                else:
                    self.labels[self.map[ensemble[0]]] = 1
        VIP_file.close()
        
        export_file = open(self.export_file, "r")
        lines = export_file.readlines()
        for line in lines:
            ensemble = line.split("\t")
            if(ensemble[0][0:4] == "ENSG"):
                if self.map[ensemble[0]] not in self.labels:
                    self.labels[self.map[ensemble[0]]] = 0
        
        export_file.close()
        return self.labels
    
        
    