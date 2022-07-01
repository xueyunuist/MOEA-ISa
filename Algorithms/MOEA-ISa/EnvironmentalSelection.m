function [Population,FrontNo,CrowdDis,P,site] = EnvironmentalSelection(Population,N,MatingPool,StateAction,LS,LC)
% The environmental selection of NSGA-II

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    if nargin>2
		[FrontNo,~] = NDSort(Population.objs,Population.cons,2*N);
		PFrontNo = FrontNo(MatingPool);  %父代的pareto等级
        PFrontNo1 = PFrontNo(1:2:end);  %父代1的pareto等级
        PFrontNo2 = PFrontNo(2:2:end);  %父代2的pareto等级
        OFrontNo = FrontNo(N+1:2*N);  %子代的pareto等级
		Dominated1 = PFrontNo1 < OFrontNo;  %判断父代1和子代pareto等级的大小判断支配关系 <为父代支配子代
        Dominated2 = PFrontNo2 < OFrontNo;  %判断父代2和子代pareto等级的大小判断支配关系
        Dominated = Dominated1 | Dominated2;  %上面所有过程是将子代依次和两个父代比较 有一个父代支配他就记为1 
		[~,uni] = unique(Population.decs,'rows');
		Population = Population(uni);  %去除重复值
		FrontNo = FrontNo(uni);
        
        NSC = zeros(LS,LC);
        NAC = zeros(LS,LC);
        for i=1:N
            NSC(StateAction(i,1),StateAction(i,2)) =NSC(StateAction(i,1),StateAction(i,2)) + ~Dominated(i);
            NAC(StateAction(i,1),StateAction(i,2)) = NAC(StateAction(i,1),StateAction(i,2)) + 1;
        end
        site = NAC~=0;
        P = NSC./NAC;
        P(isnan(P)) = 0;
        A = sort(FrontNo);
        MaxFNo = A(N);
        Next = FrontNo < MaxFNo;
    else
        % Non-dominated sorting
        [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
        Next = FrontNo < MaxFNo;
    end
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);

    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end