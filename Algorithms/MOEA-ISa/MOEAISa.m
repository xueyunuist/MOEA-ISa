function MOEAISa(Global)
% <algorithm> <M>

    %% Generate random population
    Population = InitialPop(Global.N,Global.D);
    [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Global.N);
    LC = 10;
    States = 0.05:0.05:1;
    LS = length(States);
    Ptable=zeros(LS,LC);
    alpha=0.3;   
    dataset = Global.problem.dataset;
    dataset(isnan(dataset)) = 0;
    [~,W] = relieff(dataset(:,1:end-1),dataset(:,end),20,'method','classification');

    %% Optimization
    while Global.NotTermination(Population)
        MatingPool = TournamentSelection(2,2*Global.N,FrontNo,-CrowdDis);
        Parent = Population(MatingPool);
        Offspring = Parent; 
        StateAction = zeros(Global.N,2);
        Parent = Parent.decs;
        for i=1:2:2*Global.N
            parent = Parent(i:i+1,:);
            state = sum(Parent(i,:) & Parent(i+1,:))/sum(Parent(i,:) | Parent(i+1,:));
            state = find(States >= state,1);
            P = Ptable(state,:);
            CrossoverSeq = roulette(P);
            Off  = Crossover(parent,CrossoverSeq,W); 
            Offspring((i+1)/2) = Off;
            StateAction((i+1)/2,:) =  [state CrossoverSeq];    
        end
        [Population,FrontNo,CrowdDis,p,site] = EnvironmentalSelection([Population,Offspring],Global.N,MatingPool,StateAction,LS,LC);
        Ptable(site) = (1-alpha)*Ptable(site) + alpha * p(site);
    end
end

function Population = InitialPop(N,D)
    PopDec = zeros(N,D);
    P = [0.6571 0.1664 0.0872 0.0539 0.0354];
    for i=1:N
       index = roulette(P); 
       num = randi([floor((index-1)*D/5)+1,floor(index*D/5)]);  
       PopDec(i,randperm(D,num)) = 1;
    end
    site = sum(PopDec,2)==0;
    PopDec(site,:) = rand(sum(site),D) < 0.2;
    Population = INDIVIDUAL(PopDec);
end

function Offspring = Crossover(Parent,Action,W)
    Parent1 = Parent(1,:);
    Parent2 = Parent(2,:);
    [~,D]   = size(Parent1);
    Actions = 0.1:0.1:1;
    site = Parent1 ~= Parent2;
    index = find(site);
    CrossNum = round(length(index) * Actions(Action));
    %crossover
    Offspring = Parent1 & Parent2;
    if sum(site)>1
        CrossNum = max(CrossNum,1);
        if rand < 0.5
            for i=1:CrossNum
                Index = TournamentSelection(2,1,-W(index));
                Site = index(Index);
                Offspring(Site) = 1;
                index = setdiff(index,Site);
            end
        else 
            Offspring(index(randperm(length(index),CrossNum))) = 1;
        end
    end
    %mutation
    if rand <0.5 
        if sum(Offspring)>1
            index = find(Offspring);
            Index = TournamentSelection(2,1,W(index));
            Site = index(Index);
            Offspring(Site) = 0;
        end
    else
        if sum(Offspring==0)>1
            index = find(Offspring==0);
            Index = TournamentSelection(2,1,-W(index));
            Site = index(Index);
            Offspring(Site) = 1;
        end
    end
                                   
    site = sum(Offspring,2)==0;
    Offspring(site,:) = rand(sum(site),D) < 0.2;
    Offspring = INDIVIDUAL(Offspring);
end

function strategySeq = roulette(P)
    [~,N] = size(P);
    P(P == 0) = 0.03; 
	P = P / sum(P);
    P = P * triu(ones(N)); 
    strategySeq = find(P >= rand(),1);
end
