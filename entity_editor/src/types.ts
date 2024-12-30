export interface Entity {
    entity_name: string;
    entity_type: string;
    entity_id: string;
    entity_group_id: string;
}

export interface EntityGroup {
    entity_group_id: string;
    entities: Entity[];
}

export interface EntityGroupData {
    entity_groups: EntityGroup[];
}
