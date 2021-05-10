function [ res_handle ] = summe_plotAllResults(summary_selections,methods,videoName,HOMEDATA,figureNr)
%%[res_handle] = summe_plotResult(summary_selection,videoName,HOMEDATA,figureNr)
% Evaluates a summary for video videoName and plots the results
% (where HOMEDATA points to the ground truth file)
        maxLength=0.20;        
        fontSize=16;
        symbols=['^vdosph*'];        
        colors =   [0.7 0.7 0.7;
                    0 0 0;            
                    1 0 0;
                    0 0 1;
                    1 1 0]';
                
        if exist('figureNr','var') && figureNr < 0
            res_handle=figure('Visible','off')  
        elseif exist('figureNr','var') >0
            res_handle= figure(figureNr);
        else
            res_handle= figure;
        end
        set(res_handle, 'Position', [500 500 870 550])              
        
        % Get ground truth
        load(fullfile(HOMEDATA,[videoName '.mat']));
        
        % Get automated summary score
        automated_fmeasure=[];
        automated_length=[];
        for methodIdx=1:length(methods)
            [nbSummaries,dimIdx]=min(size(summary_selections{methodIdx}));
            if dimIdx==2
                summary_selections{methodIdx}=summary_selections{methodIdx}';
            end
            automated_fmeasure{methodIdx}=[];
            automated_length{methodIdx}=[];
            if nbSummaries>1
                for selIdx=1:nbSummaries
                    curSummary=summary_selections{methodIdx}(selIdx,:);
                    [automated_fmeasure{methodIdx}(selIdx) automated_length{methodIdx}(selIdx)] = summe_evaluateSummary(curSummary,videoName,HOMEDATA);
                end
            else
                for selIdx=sort(unique(summary_selections{methodIdx}(1,:)),'ascend')
                    if selIdx>0
                        curSummary=summary_selections{methodIdx}(:)>=selIdx;
                        [automated_fmeasure{methodIdx}(end+1) automated_length{methodIdx}(end+1)] = summe_evaluateSummary(curSummary,videoName,HOMEDATA);
                    end
                end
            end
        end
        
        % Compute human scores
        for userIdx=1:size(user_score,2)
            [human_f_measures(userIdx) human_summary_length(userIdx)] = summe_evaluateSummary(user_score(:,userIdx),videoName,HOMEDATA);
        end
        avg_human_f=mean(human_f_measures);
        avg_human_len=mean(human_summary_length);
     

        plot(human_summary_length*100,human_f_measures,symbols(1),'MarkerSize',10,'Color',colors(:,1),'LineWidth', 2,'MarkerFaceColor',colors(:,1)); hold on;        
        plot(avg_human_len*100,avg_human_f,symbols(2),'MarkerSize',15,'Color',colors(:,2),'LineWidth', 2,'MarkerFaceColor',colors(:,2)); hold on;
        txtLegend={'Individual humans','Avg. human'};     
        for methodIdx=1:length(methods)
            plot(automated_length{methodIdx}*100,automated_fmeasure{methodIdx},'.--','MarkerSize',40,'Color',colors(:,2+methodIdx),'LineWidth', 2); hold on
            txtLegend{end+1}=methods{methodIdx};
        end
             
        l_handle = legend(txtLegend,'FontSize',fontSize,'Location','NorthWest');
        
        
        plot([5 5],[0 1],'--black','LineWidth', 1); hold on;
        plot([15.25 15.25],[0 1],'--black','LineWidth', 1); hold on;        
        grid minor;           
        ylbl=xlabel('summary Length [%]');
        xlbl=ylabel('mean f-measure');
        
        

        set(xlbl, 'FontName', 'Arial');
        set(xlbl, 'FontSize', fontSize) ;    
        set(ylbl, 'FontName', 'Arial');
        set(ylbl, 'FontSize', fontSize) ;    
        t_h=title(sprintf('f-measure for video %s',strrep(videoName,'_',' ')));
        set(t_h, 'FontName', 'Arial');
        set(t_h, 'FontSize', fontSize);
        set(gca,'FontSize',fontSize);        
        axis([0 maxLength*100 0 0.85]);         
     
        %%
        b2=bar([10.125],[1],10.25,'FaceColor',[0 0.3 0.0]','EdgeColor','none');
        hold off;
        %# make background bars transparent
        pch = get(b2,'child'); %# get patch objects from barseries object
        set(pch,'FaceAlpha',0.15); %# set transparency        

        
        
end

