import sys

def set_two_domain_input(images, inputs, domain, device):
    if (domain is None) or (domain == 'both'):

        # Always define data_z from inputs[0]
        data_z = inputs[0]

        # This is for adjacent slice mode.
        if not isinstance(data_z, dict):
            images.real_a = data_z.to(device, non_blocking=True)

        else:
            z1_batch = data_z['z1']
            z2_batch = data_z['z2']

            images.real_a         = z1_batch.to(device, non_blocking=True)
            images.real_a_adj     = z2_batch.to(device, non_blocking=True)
            images.real_a_names   = data_z.get('z1_name', None)
            images.real_a_adj_names = data_z.get('z2_name', None)

        images.real_b = inputs[1].to(device, non_blocking=True)

    elif domain in ['a', 0]:
        images.real_a = inputs.to(device, non_blocking=True)

    elif domain in ['b', 1]:
        images.real_b = inputs.to(device, non_blocking=True)

    else:
        raise ValueError(
            f"Unknown domain: '{domain}'."
            " Supported domains: 'a' (alias 0), 'b' (alias 1), or 'both'"
        )