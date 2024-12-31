import React, { useState } from 'react';
import { Entity, EntityGroup, EntityGroupData } from './types';
import './EntityManager.css';

const EntityManager: React.FC<{ initialData: EntityGroupData }> = ({ initialData }) => {
    const [data, setData] = useState<EntityGroupData>(initialData);
    const [selectedGroups, setSelectedGroups] = useState<string[]>([]);
    const [draggedEntity, setDraggedEntity] = useState<Entity | null>(null);

    const deleteEntity = (entityId: string, groupId: string) => {
        setData(prev => ({
            entity_groups: prev.entity_groups.map(group => {
                if (group.entity_group_id === groupId) {
                    return {
                        ...group,
                        entities: group.entities.filter(e => e.entity_id !== entityId)
                    };
                }
                return group;
            }).filter(group => group.entities.length > 0)
        }));
    };

    const deleteGroup = (groupId: string) => {
        setData(prev => ({
            entity_groups: prev.entity_groups.filter(g => g.entity_group_id !== groupId)
        }));
    };

    const mergeGroups = () => {
        if (selectedGroups.length !== 2) return;
        const [group1, group2] = selectedGroups;
        
        setData(prev => {
            const mergedEntities = [
                ...prev.entity_groups.find(g => g.entity_group_id === group1)?.entities || [],
                ...prev.entity_groups.find(g => g.entity_group_id === group2)?.entities || []
            ].map(e => ({ ...e, entity_group_id: group1 }));

            return {
                entity_groups: [
                    ...prev.entity_groups.filter(g => !selectedGroups.includes(g.entity_group_id)),
                    { entity_group_id: group1, entities: mergedEntities }
                ]
            };
        });
        setSelectedGroups([]);
    };

    const handleDragStart = (entity: Entity) => {
        setDraggedEntity(entity);
    };

    const handleDrop = (targetGroupId: string) => {
        if (!draggedEntity) return;
        
        setData(prev => ({
            entity_groups: prev.entity_groups.map(group => {
                if (group.entity_group_id === draggedEntity.entity_group_id) {
                    return {
                        ...group,
                        entities: group.entities.filter(e => e.entity_id !== draggedEntity.entity_id)
                    };
                }
                if (group.entity_group_id === targetGroupId) {
                    return {
                        ...group,
                        entities: [...group.entities, { ...draggedEntity, entity_group_id: targetGroupId }]
                    };
                }
                return group;
            }).filter(group => group.entities.length > 0)
        }));
        
        setDraggedEntity(null);
    };

    return (
        <div className="entity-manager">
            <div className="controls">
                <button 
                    onClick={mergeGroups}
                    disabled={selectedGroups.length !== 2}>
                    Merge Selected Groups
                </button>
            </div>
            <div className="groups-container">
                {data.entity_groups.map(group => (
                    <div 
                        key={group.entity_group_id}
                        className={`group ${selectedGroups.includes(group.entity_group_id) ? 'selected' : ''}`}
                        onClick={() => {
                            if (selectedGroups.includes(group.entity_group_id)) {
                                setSelectedGroups(prev => prev.filter(id => id !== group.entity_group_id));
                            } else if (selectedGroups.length < 2) {
                                setSelectedGroups(prev => [...prev, group.entity_group_id]);
                            }
                        }}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={() => handleDrop(group.entity_group_id)}
                    >
                        <div className="group-header">
                            <span>Group: {group.entity_group_id.slice(0, 8)}</span>
                            <button onClick={() => deleteGroup(group.entity_group_id)}>Delete Group</button>
                        </div>
                        <div className="entities">
                            {group.entities.map(entity => (
                                <div
                                    key={entity.entity_id}
                                    className="entity"
                                    draggable
                                    onDragStart={() => handleDragStart(entity)}
                                >
                                    <span>{entity.entity_name}</span>
                                    <button onClick={() => deleteEntity(entity.entity_id, group.entity_group_id)}>×</button>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default EntityManager;
