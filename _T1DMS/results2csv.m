function results2csv(sim_results,savedir,n_subjects)
  data = load(sim_results);
  
  mkdir([savedir]);
  mkdir(strcat("./",savedir,"/t1dms_adult"));
  mkdir(strcat("./",savedir,"/t1dms_adolescent"));
  mkdir(strcat("./",savedir,"/t1dms_child"));
   
  % loop through all subjects and extract the data
  for i=1:1:n_subjects
    subject = data.data.results(i);
    time = subject.time.signals.values(1:end-1);
    glucose = subject.sensor.signals.values(1:end-1);
    CHO = subject.CHO.signals.values(1:end-1);
    Bolus = subject.BOLUS.signals.values(1:end-1);
    
    mat = [time, glucose, CHO, Bolus];
         
    subject_class = strcat("t1dms_", subject.ID(1:end-4));
    subject_name = num2str(str2num(subject.ID(end-2:end)));
    filename = strcat(savedir,"/",subject_class,"/",subject_name,".csv");
    
    csvwrite(filename,mat);
  end
  
  zip(savedir,strcat("./",savedir,"/*"));
