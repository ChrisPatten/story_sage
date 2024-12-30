import React from 'react';
import EntityManager from './EntityManager';
import { EntityGroupData } from './types';



const initialData: EntityGroupData = {
    "entity_groups": [
        {
            "entity_group_id": "adbb226f836950d6946c78c4d29e5e83",
            "entities": [
                {
                    "entity_name": "toman head",
                    "entity_type": "entity",
                    "entity_id": "214c60d437e69ee27a239ebabc8fbbcc",
                    "entity_group_id": "adbb226f836950d6946c78c4d29e5e83"
                }
            ]
        },
        {
            "entity_group_id": "0e21c2527c15f2ef6de075d256257680",
            "entities": [
                {
                    "entity_name": "witches",
                    "entity_type": "entity",
                    "entity_id": "a6485c3cec5f6d4a00ecb6e34dc7b00f",
                    "entity_group_id": "0e21c2527c15f2ef6de075d256257680"
                }
            ]
        },
        {
            "entity_group_id": "2017cfefa959d20a21fe892f357a119d",
            "entities": [
                {
                    "entity_name": "mavra mallen",
                    "entity_type": "entity",
                    "entity_id": "eb452e654f391f71e16e71d940165b76",
                    "entity_group_id": "2017cfefa959d20a21fe892f357a119d"
                },
                {
                    "entity_name": "mavra",
                    "entity_type": "entity",
                    "entity_id": "a2c2661d26db4f794afcd163d0cad94b",
                    "entity_group_id": "2017cfefa959d20a21fe892f357a119d"
                }
            ]
        },
        {
            "entity_group_id": "52a6a8b05044298275e588476eb098fa",
            "entities": [
                {
                    "entity_name": "mara and the three foolish kings",
                    "entity_type": "entity",
                    "entity_id": "9d871cb0cccc64d238e75e3ac4f6aba6",
                    "entity_group_id": "52a6a8b05044298275e588476eb098fa"
                }
            ]
        },
        {
            "entity_group_id": "6e0c09f741012a755ec1cdd831bfafe7",
            "entities": [
                {
                    "entity_name": "seanchan woman",
                    "entity_type": "entity",
                    "entity_id": "d8a3b51519afda32215097a8ef9d0e0c",
                    "entity_group_id": "6e0c09f741012a755ec1cdd831bfafe7"
                },
                {
                    "entity_name": "seanchan",
                    "entity_type": "entity",
                    "entity_id": "803db747d4a1c8658140e2a818467918",
                    "entity_group_id": "6e0c09f741012a755ec1cdd831bfafe7"
                }
            ]
        },
        {
            "entity_group_id": "0ad7b579d469345c95a4e2fd2beedb05",
            "entities": [
                {
                    "entity_name": "floran gelb",
                    "entity_type": "entity",
                    "entity_id": "59ffdaad27b4e6246b8cd5507d344d0f",
                    "entity_group_id": "0ad7b579d469345c95a4e2fd2beedb05"
                },
                {
                    "entity_name": "gelb",
                    "entity_type": "entity",
                    "entity_id": "3195761a3be70e7cbb851a4d6fd8050d",
                    "entity_group_id": "0ad7b579d469345c95a4e2fd2beedb05"
                }
            ]
        },
        {
            "entity_group_id": "feedb2c7ce79ee453049538bcef3cbf3",
            "entities": [
                {
                    "entity_name": "leane",
                    "entity_type": "entity",
                    "entity_id": "9435ded6916866d8207e5a9b7310d4d7",
                    "entity_group_id": "feedb2c7ce79ee453049538bcef3cbf3"
                },
                {
                    "entity_name": "leane ",
                    "entity_type": "entity",
                    "entity_id": "a271ce5d7953da01afb9d047abb2686f",
                    "entity_group_id": "feedb2c7ce79ee453049538bcef3cbf3"
                }
            ]
        },
        {
            "entity_group_id": "6c93364663e4b62053a88954111b6e04",
            "entities": [
                {
                    "entity_name": "vultures",
                    "entity_type": "entity",
                    "entity_id": "0f0dace633b938adcd4d07e131c7c434",
                    "entity_group_id": "6c93364663e4b62053a88954111b6e04"
                }
            ]
        },
        {
            "entity_group_id": "ad1f6c4b165997396804e3843ba3fa80",
            "entities": [
                {
                    "entity_name": "ring of tamyrlin",
                    "entity_type": "entity",
                    "entity_id": "59feb7ed66b3e0045a4fe826569ad14f",
                    "entity_group_id": "ad1f6c4b165997396804e3843ba3fa80"
                }
            ]
        },
        {
            "entity_group_id": "6e98d575fda33f99ab939cecf5b44c57",
            "entities": [
                {
                    "entity_name": "earthworm",
                    "entity_type": "entity",
                    "entity_id": "117c19f4c30ce3ffe32e923a70b0763c",
                    "entity_group_id": "6e98d575fda33f99ab939cecf5b44c57"
                }
            ]
        },
        {
            "entity_group_id": "b81630d180620daa873dd858be6c0052",
            "entities": [
                {
                    "entity_name": "caar son of thorin",
                    "entity_type": "entity",
                    "entity_id": "aa77741317ab13b2e2af8e5008690e68",
                    "entity_group_id": "b81630d180620daa873dd858be6c0052"
                },
                {
                    "entity_name": "son",
                    "entity_type": "entity",
                    "entity_id": "8eb611bcf6aab42fbf6ad9e7b247d00a",
                    "entity_group_id": "b81630d180620daa873dd858be6c0052"
                }
            ]
        },
        {
            "entity_group_id": "2ab6ef75f435f8a1d61ad25cc76c0ee4",
            "entities": [
                {
                    "entity_name": "ryma",
                    "entity_type": "entity",
                    "entity_id": "b10bd7732fa7a6aa5a9c5c8bfc99c50a",
                    "entity_group_id": "2ab6ef75f435f8a1d61ad25cc76c0ee4"
                }
            ]
        },
        {
            "entity_group_id": "cd77279f486e9a0a148b5354869f94dc",
            "entities": [
                {
                    "entity_name": "dragons",
                    "entity_type": "entity",
                    "entity_id": "40cc113be5e8614892687019b4d9daad",
                    "entity_group_id": "cd77279f486e9a0a148b5354869f94dc"
                },
                {
                    "entity_name": "false dragons",
                    "entity_type": "entity",
                    "entity_id": "53df32f169e61c28797e5821af323645",
                    "entity_group_id": "cd77279f486e9a0a148b5354869f94dc"
                },
                {
                    "entity_name": "dragons fang",
                    "entity_type": "entity",
                    "entity_id": "f2c09d81b880d2743f326e34c7b74243",
                    "entity_group_id": "cd77279f486e9a0a148b5354869f94dc"
                }
            ]
        },
        {
            "entity_group_id": "aad8b851879e9e66c2fb53083cf5ed0c",
            "entities": [
                {
                    "entity_name": "rhyagelle",
                    "entity_type": "entity",
                    "entity_id": "cdb54fc9a8962e3f5a3dcab2d47d734f",
                    "entity_group_id": "aad8b851879e9e66c2fb53083cf5ed0c"
                }
            ]
        },
        {
            "entity_group_id": "4361e940fe77b19ab4c367196ee55f4a",
            "entities": [
                {
                    "entity_name": "nephews",
                    "entity_type": "entity",
                    "entity_id": "7d4cf4c30d9bc5dae3cae0241c186ffc",
                    "entity_group_id": "4361e940fe77b19ab4c367196ee55f4a"
                }
            ]
        },
        {
            "entity_group_id": "455c403004060d3f31dce0b5380e6761",
            "entities": [
                {
                    "entity_name": "gleeman",
                    "entity_type": "entity",
                    "entity_id": "6112d6234d8325cc06bc608e62663901",
                    "entity_group_id": "455c403004060d3f31dce0b5380e6761"
                },
                {
                    "entity_name": "gleeman\u2019s tale",
                    "entity_type": "entity",
                    "entity_id": "b1bcba54b6cd8ed46e7446c84f92ef72",
                    "entity_group_id": "455c403004060d3f31dce0b5380e6761"
                },
                {
                    "entity_name": "the gleeman",
                    "entity_type": "entity",
                    "entity_id": "3f63ac11cef0130ded90c28594223795",
                    "entity_group_id": "455c403004060d3f31dce0b5380e6761"
                },
                {
                    "entity_name": "master gleeman",
                    "entity_type": "entity",
                    "entity_id": "dce7b6900512e608971eeb42305bf6dc",
                    "entity_group_id": "455c403004060d3f31dce0b5380e6761"
                }
            ]
        },
        {
            "entity_group_id": "95b49279a4daa54f62695742f52087e4",
            "entities": [
                {
                    "entity_name": "white ajah",
                    "entity_type": "entity",
                    "entity_id": "731c1fc8565bb8939dd79735220f62cc",
                    "entity_group_id": "95b49279a4daa54f62695742f52087e4"
                },
                {
                    "entity_name": "the black ajah",
                    "entity_type": "entity",
                    "entity_id": "bb6a67c5c0d55974fc75275e84d9e780",
                    "entity_group_id": "95b49279a4daa54f62695742f52087e4"
                },
                {
                    "entity_name": "black ajah",
                    "entity_type": "entity",
                    "entity_id": "6f9e621164f3e3195bf743596c5eb44a",
                    "entity_group_id": "95b49279a4daa54f62695742f52087e4"
                },
                {
                    "entity_name": "red ajah",
                    "entity_type": "entity",
                    "entity_id": "853fbfe8273ada2b8c22a416125650e8",
                    "entity_group_id": "95b49279a4daa54f62695742f52087e4"
                },
                {
                    "entity_name": "ajah",
                    "entity_type": "entity",
                    "entity_id": "1740316762551a665702c51a69ca2027",
                    "entity_group_id": "95b49279a4daa54f62695742f52087e4"
                }
            ]
        },
        {
            "entity_group_id": "09f5acada0b159c810292187231b1ad5",
            "entities": [
                {
                    "entity_name": "guardsmen",
                    "entity_type": "entity",
                    "entity_id": "4ce273d6225ce1dff272cf07e6d52fe3",
                    "entity_group_id": "09f5acada0b159c810292187231b1ad5"
                }
            ]
        },
        {
            "entity_group_id": "b39c08e5377335bc32cd17998fc0a22b",
            "entities": [
                {
                    "entity_name": "prisoners",
                    "entity_type": "entity",
                    "entity_id": "859ba5a3952bfb50e64ec197e2ffab55",
                    "entity_group_id": "b39c08e5377335bc32cd17998fc0a22b"
                }
            ]
        },
        {
            "entity_group_id": "47d2ccc4d7ff9ecb5f11154303f0fb29",
            "entities": [
                {
                    "entity_name": "i wish",
                    "entity_type": "entity",
                    "entity_id": "2d0382fe72341997b18499a6ded2432d",
                    "entity_group_id": "47d2ccc4d7ff9ecb5f11154303f0fb29"
                }
            ]
        },
        {
            "entity_group_id": "484e8f1d17124653ed2b7e94a5c24533",
            "entities": [
                {
                    "entity_name": "color",
                    "entity_type": "entity",
                    "entity_id": "0e9baeaef1598aac4a180e8be19e35da",
                    "entity_group_id": "484e8f1d17124653ed2b7e94a5c24533"
                }
            ]
        },
        {
            "entity_group_id": "284e03aeab6e8ad3ef318d2792bf780f",
            "entities": [
                {
                    "entity_name": "sara",
                    "entity_type": "entity",
                    "entity_id": "3f048bead445d9a2977d75fdb2bbce33",
                    "entity_group_id": "284e03aeab6e8ad3ef318d2792bf780f"
                },
                {
                    "entity_name": "darling sara",
                    "entity_type": "entity",
                    "entity_id": "6e707e0392807420338abb31d6175402",
                    "entity_group_id": "284e03aeab6e8ad3ef318d2792bf780f"
                }
            ]
        },
        {
            "entity_group_id": "1bf714fe74ecc7ab20f59063e87c1a73",
            "entities": [
                {
                    "entity_name": "wind",
                    "entity_type": "entity",
                    "entity_id": "9e059a24a6c255306ff2684b225e83e6",
                    "entity_group_id": "1bf714fe74ecc7ab20f59063e87c1a73"
                },
                {
                    "entity_name": "black wind",
                    "entity_type": "entity",
                    "entity_id": "0371520ac4f2ef5c15cb6bdb8cd85153",
                    "entity_group_id": "1bf714fe74ecc7ab20f59063e87c1a73"
                }
            ]
        },
        {
            "entity_group_id": "37f8a56b5c57454f33d92a522e5382a2",
            "entities": [
                {
                    "entity_name": "gulls",
                    "entity_type": "entity",
                    "entity_id": "410a0cfc39f9e96173e2f6a90a8bb256",
                    "entity_group_id": "37f8a56b5c57454f33d92a522e5382a2"
                }
            ]
        },
        {
            "entity_group_id": "1d385648da5cecefe3f8604405fa2109",
            "entities": [
                {
                    "entity_name": "matrim",
                    "entity_type": "entity",
                    "entity_id": "56be4bad2e8b10a239fab8fef4bfd081",
                    "entity_group_id": "1d385648da5cecefe3f8604405fa2109"
                },
                {
                    "entity_name": "matrim cauthon",
                    "entity_type": "entity",
                    "entity_id": "b24b23642ad67dd6b9a9128d24b35d33",
                    "entity_group_id": "1d385648da5cecefe3f8604405fa2109"
                }
            ]
        },
        {
            "entity_group_id": "2928d4c1370a9df56b07b43039b5d388",
            "entities": [
                {
                    "entity_name": "thom",
                    "entity_type": "entity",
                    "entity_id": "2f597ae34654ca313da3c46b9b03902f",
                    "entity_group_id": "2928d4c1370a9df56b07b43039b5d388"
                },
                {
                    "entity_name": "thom merrilin",
                    "entity_type": "entity",
                    "entity_id": "9a4095ba3a517894252fbe863f42c8d7",
                    "entity_group_id": "2928d4c1370a9df56b07b43039b5d388"
                }
            ]
        }
    ]
};

const App: React.FC = () => {
    return (
        <div>
            <h1>Entity Group Manager</h1>
            <EntityManager initialData={initialData} />
        </div>
    );
};

export default App;
