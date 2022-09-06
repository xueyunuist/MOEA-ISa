for i=1:15
	FS = sprintf('FS%d',i);
	for r = 1 : 30 
		main('-algorithm',@MOEAISa,'-problem',str2func(FS1),'-evaluation',10000,'-save',10,'-N',100,'-M',2); 
	end 
end
