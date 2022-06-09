function [Gbest_val,fitness_record,pos]= EOPSO(fhd,MaxFES,swarm_size,dimension,LB,UB,varargin)
% Gbest_val: global best position's fitness
% Pbest_val: personal best position's fitness
% fitness_record:  storage convergence curve
% MaxFES Maximum: fitness evaluation times
% swarm_size: population size
% dimension: search dimension
% LB/UP: lower/upper search boundry
% pos: position
% pos: velocity
% inw: inertia weight
rand('state',sum(100*clock));

fitness_record=zeros(1,MaxFES);
max_iteration=ceil(MaxFES/swarm_size);
vel_max=(UB-LB)/2;
vel_min=-vel_max;
pos=rand(swarm_size,dimension).*(UB-LB)+LB;
vel=rand(swarm_size,dimension).*(vel_max-vel_min)+vel_min;
fitness=zeros(swarm_size,1);
for i=1:swarm_size
    fitness(i)=feval(fhd,pos(i,:)',varargin{:});
end
fitcount=swarm_size;
fitness_record(1:fitcount)=min(fitness);
[min_val,min_index]=min(fitness);
Gbest=pos(min_index,:);
Gbest_val=min_val;
Pbest=pos;
Pbest_val=fitness;
count=zeros(1,swarm_size);
c=1.49445;
iteration=1;
Rmax=0.5;
Rmin=0.4;
G=7;

while iteration<=max_iteration & fitcount<=MaxFES
    inw=0.9-0.5*(iteration/max_iteration);
    iteration=iteration+1;
    K=ceil(swarm_size*Rmax-(Rmax-Rmin)*iteration/max_iteration*swarm_size);
    [~,index]=sort(fitness);
    OO=zeros(1,dimension);
    for i=1:K
        if fitness(index(i))>0
            f(i)=1/(fitness(index(i))+1);
        else
            f(i)=1+abs(fitness(index(i)));
        end
    end
    
    elite=zeros(1,K);
    elite(1:K)=index(1:K);
    ordinary=zeros(1,swarm_size-K);
    ordinary(1:swarm_size-K)=index(K+1:swarm_size);
    
    for i=1:K
        OO=OO+f(i)/sum(f)*Pbest(index(i),:);
    end
    
    for i=1:swarm_size
        ranknum=index(i);
        if ranknum<=K
            vel(i,:)=inw.*vel(i,:)+c*rand(1,dimension).*(Pbest(i,:)-pos(i,:));
        else
            vel(i,:)=inw.*vel(i,:)+c*rand(1,dimension).*(OO-pos(i,:));
        end
        vel(i,vel(i,:)>vel_max) = vel_max;
        vel(i,vel(i,:)<vel_min) = vel_min;
        pos(i,:) = pos(i,:) + vel(i,:);
        pos(i,pos(i,:)>UB) = UB;
        pos(i,pos(i,:)<LB) = LB;
        fitness(i)=feval(fhd,pos(i,:)',varargin{:});
        fitcount=fitcount+1;
        fitness_record(fitcount)=min(fitness_record(fitcount-1),fitness(i));
        Pbest(i,:)=(fitness(i)<Pbest_val(i))*pos(i,:)+(fitness(i)>=Pbest_val(i))*Pbest(i,:);
        Pbest_val(i)=(fitness(i)<Pbest_val(i))*fitness(i)+(fitness(i)>=Pbest_val(i))*Pbest_val(i);
        count(i)=(fitness(i)>Pbest_val(i))+(fitness(i)>Pbest_val(i))*count(i);
        Gbest=(fitness(i)<Gbest_val)*pos(i,:)+(fitness(i)>=Gbest_val)*Gbest;
        Gbest_val=(fitness(i)<Gbest_val)*fitness(i)+(fitness(i)>=Gbest_val)*Gbest_val;
        if count(i)>=G
            count(i)=0;
            r1=(ranknum<=K)*ordinary(randperm(swarm_size-K,1))+(1-(ranknum<=K))*elite(randperm(K,1));
            RR=rand(1,dimension);
            PX=RR.*Pbest(i,:)+(1-RR).*Pbest(r1,:)+2*(rand(1,dimension)-0.5).*(Pbest(i,:)-Pbest(r1,:));
            PX_fit=feval(fhd,PX',varargin{:});
            fitcount=fitcount+1;
            fitness_record(fitcount)=min(fitness_record(fitcount-1),PX_fit);
            pos(i,:)=PX*(PX_fit<fitness(i))+pos(i,:)*(1-(PX_fit<fitness(i)));
            fitness(i)=PX_fit*(PX_fit<fitness(i))+fitness(i,:)*(1-(PX_fit<fitness(i)));
        end
        vel(i,:)= (abs(vel(i,:))>10^(-4)).* vel(i,:)+(1-(abs(vel(i,:))>10^(-4))).*normrnd(0,1,[1,dimension]);
        
    end
    
    if fitcount>=MaxFES
        break;
    end
    if (iteration==max_iteration) & (fitcount<MaxFES)
        iteration=iteration-1;
    end
end
fitness_record=fitness_record(1:MaxFES);
Gbest_val=fitness_record(1,MaxFES);
end

