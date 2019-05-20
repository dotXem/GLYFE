function results2csv(data,savedir)
  % create folder if it doesn't exist
  Folder=pwd;
  [PathStr,FolderName]=fileparts(Folder);
  DataFolder=[savedir];
  mkdir(DataFolder);
  
  % loop through all subjects and extract the data
  for i=1:1:30
    subject = data.data.results(i);
    time = subject.time.signals.values;
    glucose = subject.G.signals.values;
    CHO = subject.CHO.signals.values;
    Bolus = subject.BOLUS.signals.values;
    
    mat = [time, glucose, CHO, Bolus];
         
    filename = strcat(savedir,"/",subject.ID,".csv");
    
    csvwrite(filename,mat);
  end
  
  zip(savedir,strcat("./",savedir,"/*"))
